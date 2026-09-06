"""
evaluate_test.py
Loads the best checkpoint and reports per-class AUROC + accuracy
(at a 0.5 probability threshold) on the held-out test set.
"""
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from dataset import ChestXrayDataset, CLASS_COLUMNS
from model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    test_ds = ChestXrayDataset(f"../manifest_test.csv",
    f"../processed_data/test", "test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    model = build_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=DEVICE))
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            probs = torch.sigmoid(model(images))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    preds = (all_probs >= 0.5).astype(int)
    print(f"{'Class':<18s}{'AUROC':>10s}{'Accuracy':>12s}")

    for i, cls in enumerate(CLASS_COLUMNS):
        if len(np.unique(all_labels[:, i])) < 2:
            auroc_str = "n/a"
        else:
            auroc_str = f"{roc_auc_score(all_labels[:, i], all_probs[:, i]):.4f}"
        acc = accuracy_score(all_labels[:, i], preds[:, i])
        print(f"{cls:<18s}{auroc_str:>10s}{acc:>12.4f}")

if __name__ == "__main__":
    main()