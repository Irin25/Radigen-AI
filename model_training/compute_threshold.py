"""
compute_thresholds.py

Computes per-class decision thresholds on the validation set.

For each class:
    1. Look for a threshold where BOTH sensitivity >= MIN_SENS and
       specificity >= MIN_SPEC (default 0.80 / 0.80). If several exist, pick
       the one with the largest sensitivity + specificity.
    2. If no such threshold exists (the model is not good enough for that
       class), fall back to the highest-sensitivity threshold that still
       reaches FALLBACK_SPECIFICITY, and mark the class "below target".

Output:
    thresholds.json   <- used by gradcam++_explain.py / the website
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

# Target for both metrics
MIN_SENS = 0.78
MIN_SPEC = 0.83

# Used only for classes that cannot reach both targets.
# Higher = fewer false alarms but lower sensitivity.
FALLBACK_SPECIFICITY = 0.80

# Optional per-class overrides of the fallback specificity, e.g. {"Pneumonia": 0.90}
FALLBACK_OVERRIDES = {}


def get_val_probs_labels(model, loader):
    model.eval()
    all_probs, all_labels, all_masks = [], [], []
    with torch.no_grad():
        for images, labels, masks in loader:
            images = images.to(DEVICE)
            with autocast(device_type=DEVICE):
                probs = torch.sigmoid(model(images))
            all_probs.append(probs.float().cpu().numpy())
            all_labels.append(labels.numpy())
            all_masks.append(masks.numpy())
    return (np.concatenate(all_probs),
            np.concatenate(all_labels),
            np.concatenate(all_masks))


def _clean(t):
    # roc_curve's first threshold can be inf
    return float(t) if np.isfinite(t) else 1.0


def threshold_for_both(labels_i, probs_i, min_sens, min_spec):
    """Returns (threshold, sensitivity, specificity) or None if infeasible."""
    fpr, tpr, thr = roc_curve(labels_i, probs_i)
    spec = 1.0 - fpr
    ok = np.where((tpr >= min_sens) & (spec >= min_spec))[0]
    if len(ok) == 0:
        return None
    idx = ok[np.argmax(tpr[ok] + spec[ok])]
    return _clean(thr[idx]), float(tpr[idx]), float(spec[idx])


def threshold_for_specificity(labels_i, probs_i, target_spec):
    """Highest-sensitivity threshold with specificity >= target_spec."""
    fpr, tpr, thr = roc_curve(labels_i, probs_i)
    valid = np.where(fpr <= (1.0 - target_spec))[0]   # never empty: fpr[0] == 0
    idx = valid[np.argmax(tpr[valid])]
    return _clean(thr[idx]), float(tpr[idx]), float(1.0 - fpr[idx])


def compute_thresholds(probs, labels, masks):
    thresholds = {}

    print(f"\nTarget: sensitivity >= {MIN_SENS:.2f} and specificity >= {MIN_SPEC:.2f}")
    print(f"\n{'Class':<18s}{'Threshold':>10s}{'Sens':>8s}{'Spec':>8s}   Status")
    print("-" * 62)

    for i, cls in enumerate(CLASS_COLUMNS):
        valid = masks[:, i] == 1
        labels_i, probs_i = labels[valid, i], probs[valid, i]

        if len(np.unique(labels_i)) < 2:
            thresholds[cls] = 0.5
            print(f"{cls:<18s}{'0.500':>10s}{'N/A':>8s}{'N/A':>8s}   not enough data")
            continue

        result = threshold_for_both(labels_i, probs_i, MIN_SENS, MIN_SPEC)
        if result is not None:
            t, se, sp = result
            status = "OK (meets target)"
        else:
            fb = FALLBACK_OVERRIDES.get(cls, FALLBACK_SPECIFICITY)
            t, se, sp = threshold_for_specificity(labels_i, probs_i, fb)
            status = f"below target (fallback spec>={fb:.2f})"

        thresholds[cls] = t
        print(f"{cls:<18s}{t:>10.3f}{se:>8.2f}{sp:>8.2f}   {status}")

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

    nih_p, nih_l, nih_m = get_val_probs_labels(model, nih_loader)
    chx_p, chx_l, chx_m = get_val_probs_labels(model, chexpert_loader)

    probs = np.concatenate([nih_p, chx_p])
    labels = np.concatenate([nih_l, chx_l])
    masks = np.concatenate([nih_m, chx_m])

    thresholds = compute_thresholds(probs, labels, masks)

    # Average number of pathologies flagged per image (excluding No Finding)
    path_idx = [i for i, c in enumerate(CLASS_COLUMNS) if c != "No Finding"]
    t_path = np.array([thresholds[CLASS_COLUMNS[i]] for i in path_idx])
    flagged = (probs[:, path_idx] >= t_path).sum(axis=1).mean()
    print(f"\nAverage pathologies flagged per image: {flagged:.2f}")

    with open("thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    print("Saved thresholds to thresholds.json")


if __name__ == "__main__":
    main()