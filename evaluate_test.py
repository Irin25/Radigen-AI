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
    test_ds_nih = ChestXrayDataset(f"../nih_common_test.csv",
    f"../processed_data/test", f"../processed_data_chexpert/test", "test")
    test_ds_chexpert = ChestXrayDataset(f"../chexpert_common_test.csv",
    f"../processed_data/test", f"../processed_data_chexpert/test", "test")
    test_loader_nih = DataLoader(test_ds_nih, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader_chexpert = DataLoader(test_ds_chexpert, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    model = build_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=DEVICE))
    model.eval()

    # ---------- NIH TEST ----------
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels, masks in test_loader_nih:
            images = images.to(DEVICE)

            probs = torch.sigmoid(model(images))

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    preds = (all_probs >= 0.5).astype(int)

    print("\nNIH TEST RESULTS")
    print(f"{'Class':<18s}{'AUROC':>10s}{'Accuracy':>12s}")

    for i, cls in enumerate(CLASS_COLUMNS):
        if len(np.unique(all_labels[:, i])) < 2:
            auroc_str = "n/a"
        else:
            auroc_str = f"{roc_auc_score(all_labels[:, i], all_probs[:, i]):.4f}"

        acc = accuracy_score(all_labels[:, i], preds[:, i])

        print(f"{cls:<18s}{auroc_str:>10s}{acc:>12.4f}")

    # ---------- CHEXPERT TEST ----------
    all_probs, all_labels, all_masks = [], [], []

    with torch.no_grad():
        for images, labels, masks in test_loader_chexpert:
            images = images.to(DEVICE)

            probs = torch.sigmoid(model(images))

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_masks.append(masks.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_masks = np.concatenate(all_masks)

    preds = (all_probs >= 0.5).astype(int)

    print("\nCHEXPERT TEST RESULTS")
    print(f"{'Class':<18s}{'AUROC':>10s}{'Accuracy':>12s}")

    for i, cls in enumerate(CLASS_COLUMNS):

        valid = all_masks[:, i] > 0

        y_true = all_labels[valid, i]
        y_prob = all_probs[valid, i]
        y_pred = (y_prob >= 0.5).astype(int)

        if len(np.unique(y_true)) < 2:
            auroc_str = "n/a"
        else:
            auroc_str = f"{roc_auc_score(y_true, y_prob):.4f}"

        acc = accuracy_score(y_true, y_pred)

        print(f"{cls:<18s}{auroc_str:>10s}{acc:>12.4f}")

if __name__ == "__main__":
    main()