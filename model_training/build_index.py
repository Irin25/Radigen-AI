"""
build_index.py

Reads medical text files from the corpus folder,
creates one Document for each [Class | Section],
converts them into embeddings,
and saves a FAISS index for later retrieval.
"""

import os
import re
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

CORPUS_DIR = "corpus"
INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_corpus(corpus_dir):
    documents = []
    for fname in os.listdir(corpus_dir):
        if not fname.endswith(".txt"):
            continue
        file_path = os.path.join(corpus_dir, fname)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        # Split the file whenever a new [Class | Section] heading appears
        sections = re.split(r"\n(?=\[.*?\|.*?\])", text)
        for section in sections:
            section = section.strip()
            if not section:
                continue
            # Separate the heading from the actual text
            lines = section.split("\n", 1)
            if len(lines) != 2:
                continue
            heading = lines[0].strip()
            content = lines[1].strip()
            # Remove [ and ]
            heading = heading.strip("[]")
            # Split:
            # "Atelectasis | Definition"
            # into:
            # "Atelectasis"
            # "Definition"
            if "|" in heading:
                class_name, section_name = [
                    x.strip() for x in heading.split("|", 1)
                ]
            else:
                # Example: "Disclaimer"
                class_name = "General"
                section_name = heading
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "class": class_name,
                        "section": section_name,
                        "source": fname
                    }
                )
            )
    return documents


def main():
    # Load the medical corpus
    documents = load_corpus(CORPUS_DIR)
    print(f"Loaded {len(documents)} sections from {CORPUS_DIR}/")
    # Create the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    # Convert documents into embeddings and create FAISS index
    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )
    # Save FAISS index
    vectorstore.save_local(INDEX_PATH)
    print(f"FAISS index saved to {INDEX_PATH}/")


if __name__ == "__main__":
    main()

