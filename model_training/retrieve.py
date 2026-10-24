"""
retrieve.py
Loads the saved FAISS index and retrieves the most relevant passages
for a given predicted disease class. 
"""
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings = None
_vectorstore = None

def _load_index():
    global _embeddings, _vectorstore
    if _vectorstore is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = FAISS.load_local(
        INDEX_PATH, _embeddings, allow_dangerous_deserialization=True
        )
    return _vectorstore

def retrieve_for_class(class_name):
    vectorstore = _load_index()

    results = [
        doc
        for doc in vectorstore.docstore._dict.values()
        if doc.metadata.get("class") == class_name
        and doc.metadata.get("section") in ["Definition", "X-ray appearance"]
    ]

    # Keep the order: Definition first, X-ray appearance second
    section_order = {
        "Definition": 0,
        "X-ray appearance": 1
    }

    results.sort(key=lambda doc: section_order[doc.metadata["section"]])

    combined_text = " ".join(doc.page_content for doc in results) 
    return combined_text

MAX_DIST = 1.2   # FAISS squared-L2 cut-off for "relevant enough"; lower = stricter. Tune on your corpus.


def answer_question(question, k=2):
    """Chatbot: answers only from the knowledge base (retrieval, no free generation)."""
    hits = [(d, s) for d, s in _load_index().similarity_search_with_score(question, k=k) if s <= MAX_DIST]
    if not hits:
        return {"answer": "I couldn't find that in the knowledge base. Try asking about a finding, the heatmap, or the model's limits.", "sources": []}
    return {"answer": " ".join(d.page_content for d, _ in hits),
            "sources": [f'{d.metadata["class"]} · {d.metadata["section"]}' for d, _ in hits]}
