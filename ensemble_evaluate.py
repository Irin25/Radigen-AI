import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from dataset import ChestXrayDataset, CLASS_COLUMNS
from model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the peak-performing epochs around the Epoch 5 maximum
CHECKPOINT_PATHS = [
    "checkpoints/best_model.pt",  # Epoch 5 (Val AUROC: 0.8425)
    "checkpoints/epoch_02.pt",     # Val AUROC: 0.8403
    "checkpoints/epoch_03.pt",     # Val AUROC: 0.8417
]

def get_model_probabilities(checkpoint_path, loader):
    model = build_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    all_probs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)
            probs = torch.sigmoid(model(images))
            all_probs.append(probs.cpu().numpy())
            
    return np.concatenate(all_probs)

def main():
    test_ds = ChestXrayDataset("../manifest_test.csv", "../processed_data/test", "test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Collect ground truth labels
    all_labels = []
    for _, labels in test_loader:
        all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels)

    # Accumulate predictions across all checkpoint models
    accumulated_probs = np.zeros_like(all_labels, dtype=np.float64)
    
    for path in CHECKPOINT_PATHS:
        print(f"Generating predictions from {path}...")
        probs = get_model_probabilities(path, test_loader)
        accumulated_probs += probs

    # Average the predicted probabilities across the ensemble
    ensemble_probs = accumulated_probs / len(CHECKPOINT_PATHS)
    preds = (ensemble_probs >= 0.5).astype(int)

    print("\n--- ENSEMBLE TEST RESULTS ---")
    print(f"{'Class':<18s}{'AUROC':>10s}{'Accuracy':>12s}")

    for i, cls in enumerate(CLASS_COLUMNS):
        if len(np.unique(all_labels[:, i])) < 2:
            auroc_str = "n/a"
        else:
            auroc_str = f"{roc_auc_score(all_labels[:, i], ensemble_probs[:, i]):.4f}"
        acc = accuracy_score(all_labels[:, i], preds[:, i])
        print(f"{cls:<18s}{auroc_str:>10s}{acc:>12.4f}")

if __name__ == "__main__":
    main()