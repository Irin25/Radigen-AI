"""
train.py
Fine-tunes Swin-Tiny on the 8-class multi-label chest X-ray dataset,
using fp16 mixed precision and gradient accumulation to fit a 6GB GPU.

Modified to use Focal Loss (instead of weighted BCE) and early stopping,
to target the Pneumonia AUROC weakness and the overfitting seen after
epoch 5 in the original run.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
import numpy as np
from dataset import ChestXrayDataset, CLASS_COLUMNS
from model import build_model
from compute_pos_weight import compute_pos_weight

BATCH_SIZE = 32 # physical batch size that fits in 6GB VRAM
ACCUM_STEPS = 1 # effective batch size = 32 * 1 = 32
NUM_EPOCHS = 15
LEARNING_RATE = 3e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Focal loss config ----
USE_ALPHA = False   # start with alpha=None (focal term alone). Set True to also
                     # apply pos_weight-style class balancing on top of focusing.
GAMMA = 2.0          # focusing parameter; try 1.5 if training looks unstable,
                     # 3.0 if Pneumonia still isn't moving after this run.

# ---- Early stopping config ----
PATIENCE = 3         # stop if val mean AUROC doesn't improve for this many epochs


class FocalLoss(nn.Module):
    """
    Minimal multi-label focal loss, drop-in replacement for
    nn.BCEWithLogitsLoss(pos_weight=...).
    """
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # tensor of shape [num_classes] or None
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma

        loss = focal_term * bce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha * targets + (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad(): # no gradient descent done during evaluation , just finding its accuracy before training
        for images, labels in loader:
            images = images.to(DEVICE)
            with autocast(device_type=DEVICE): # automatically manages which operations use which precision
                logits = model(images)
                probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Per-class AUROC; skip a class if it has only one label value present otherwise it wont be able to calculate AUROC
    # AUROC evaluates how well the model's probability scores/ranking separate 
    # positive cases from negative cases across different decision thresholds.
    aurocs = {}
    for i, cls in enumerate(CLASS_COLUMNS):
        if len(np.unique(all_labels[:, i])) < 2:
            continue
        aurocs[cls] = roc_auc_score(all_labels[:, i], all_probs[:, i])
    mean_auroc = np.mean(list(aurocs.values())) if aurocs else float("nan")
    return aurocs, mean_auroc

def main():
    train_ds = ChestXrayDataset(f"../manifest_train.csv",
    f"../processed_data/train", "train")
    val_ds = ChestXrayDataset(f"../manifest_val.csv",
    f"../processed_data/val", "val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True) # 4 CPU workers prepare/load batch when CPU is working with another batch

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True, persistent_workers=True)

    model = build_model().to(DEVICE)

    # pos_weight is now optional -- only computed/used if USE_ALPHA=True,
    # since focal loss's gamma term already targets the same hard/rare-class
    # problem pos_weight was compensating for. Stacking both can overcorrect.
    alpha = compute_pos_weight(f"../manifest_train.csv").to(DEVICE) if USE_ALPHA else None

    criterion = FocalLoss(gamma=GAMMA, alpha=alpha)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS
    )

    scaler = GradScaler(device=DEVICE) # helps prevent very small FP16 gradients from becoming numerically useless during training.
    best_auroc = 0.0
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, (images, labels) in enumerate(train_loader): # step indicates which batch currently on
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            with autocast(device_type=DEVICE):
                logits = model(images)
                loss = criterion(logits, labels) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer) # changing models weights only after the accumulation steps are finished
                scaler.update()
                optimizer.zero_grad()
            running_loss += loss.item() * ACCUM_STEPS

        if len(train_loader) % ACCUM_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_loss = running_loss / len(train_loader)

        aurocs, mean_auroc = evaluate(model, val_loader)

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - "
        f"LR: {optimizer.param_groups[0]['lr']:.2e} - "
        f"train loss: {avg_loss:.4f} - "
        f"val mean AUROC: {mean_auroc:.4f}")

        for cls, score in aurocs.items():
            print(f" {cls:<18s} AUROC: {score:.4f}")

        os.makedirs("checkpoints", exist_ok=True)

        epoch_ckpt_path = f"checkpoints/epoch_{epoch+1:02d}.pt"
        torch.save(model.state_dict(), epoch_ckpt_path)

        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f" -> New best model saved (mean AUROC {best_auroc:.4f})")
        else:
            epochs_without_improvement += 1
            print(f" -> No improvement for {epochs_without_improvement}/{PATIENCE} epoch(s)")

        scheduler.step()

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1} "
                  f"(no improvement for {PATIENCE} consecutive epochs).")
            break

    print(f"\nTraining complete. Best validation mean AUROC: {best_auroc:.4f}")

if __name__ == "__main__":
    main()