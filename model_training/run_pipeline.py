import sys
from gradcam_explain import run_gradcam
from retrieve import retrieve_for_class

def run_full_demo(image_path):

    heatmap_path, positive_classes = run_gradcam(
        image_path,
        all_positive=True
    )

    all_passages = []

    for class_name in positive_classes:
        passages = retrieve_for_class(class_name)
        all_passages.append(passages)

    print("\nRetrieved knowledge:")

    for p in all_passages:
        print(f"- {p}")

    return { "heatmap_path": heatmap_path, "retrieved_passages": all_passages }

if __name__ == "__main__":
    if len(sys.argv) < 2: 
        print("Usage: python run_full_demo.py <image_path>") 
        sys.exit(1) 
    image_path = sys.argv[1] 
    run_full_demo(image_path)