"""
compute_thresholds.py
Computes the optimal per-class decision threshold (Youden's J) on the
validation set, using the same model/checkpoint as training.
Saves thresholds to thresholds.json for gradcam_explain.py to use.
"""

import json
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve

from dataset import ChestXrayDataset, CLASS_COLUMNS
from model import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "checkpoints/best_model.pt"
BATCH_SIZE = 32


def get_val_probs_labels(model, loader):
    model.eval()
    all_probs, all_labels, all_masks = [], [], []
    with torch.no_grad():
        for images, labels, masks in loader:
            images = images.to(DEVICE)
            with autocast(device_type=DEVICE):
                logits = model(images)
                probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_masks.append(masks.numpy())
    return (np.concatenate(all_probs),
            np.concatenate(all_labels),
            np.concatenate(all_masks))


def compute_optimal_thresholds(probs, labels, masks):
    thresholds = {}
    for i, cls in enumerate(CLASS_COLUMNS):
        valid = masks[:, i] == 1
        labels_i = labels[valid, i]
        probs_i = probs[valid, i]

        if len(np.unique(labels_i)) < 2:
            thresholds[cls] = 0.5  # fallback, not enough data to optimize
            continue

        fpr, tpr, thresh_vals = roc_curve(labels_i, probs_i)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        thresholds[cls] = float(thresh_vals[best_idx])

    return thresholds


def main():
    nih_val_ds = ChestXrayDataset(
        "../nih_common_val.csv", "../processed_data/val",
        "../processed_data_chexpert/val", "val"
    )
    chexpert_val_ds = ChestXrayDataset(
        "../chexpert_common_val.csv", "../processed_data/val",
        "../processed_data_chexpert/val", "val"
    )

    nih_loader = DataLoader(nih_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    chexpert_loader = DataLoader(chexpert_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))

    # Combine both val sets so thresholds reflect overall behavior
    nih_probs, nih_labels, nih_masks = get_val_probs_labels(model, nih_loader)
    chex_probs, chex_labels, chex_masks = get_val_probs_labels(model, chexpert_loader)

    all_probs = np.concatenate([nih_probs, chex_probs])
    all_labels = np.concatenate([nih_labels, chex_labels])
    all_masks = np.concatenate([nih_masks, chex_masks])

    thresholds = compute_optimal_thresholds(all_probs, all_labels, all_masks)

    print("\nOptimal per-class thresholds (Youden's J):")
    for cls, t in thresholds.items():
        print(f"  {cls:<18s} {t:.3f}")

    with open("thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print("\nSaved to thresholds.json")


if __name__ == "__main__":
    main()