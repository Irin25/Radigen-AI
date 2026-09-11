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

BATCH_SIZE = 32 # physical batch size that fits in 6GB VRAM
ACCUM_STEPS = 1 # effective batch size = 32 * 1 = 32
NUM_EPOCHS = 15
LEARNING_RATE = 3e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Focal loss config ----
GAMMA = 1.5          # focusing parameter
                   # gamma controls how much the loss focuses on difficult/misclassified examples.

# ---- Early stopping config ----
PATIENCE = 2         # stop if val mean AUROC doesn't improve for this many epochs


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, class_weights=None,  reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, logits, targets, masks):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma

        loss = focal_term * bce_loss
        # Ignore uncertain CheXpert labels
        loss = loss * masks

        if self.class_weights is not None:
            loss = loss * self.class_weights

        if self.reduction == "mean":
             return loss.sum() / masks.sum().clamp(min=1.0)
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def evaluate(model, loader):
    model.eval()
    all_probs, all_labels, all_masks = [], [], []
    with torch.no_grad(): # no gradient descent done during evaluation , just finding its accuracy before training
        for images, labels, masks in loader:
            images = images.to(DEVICE)
            with autocast(device_type=DEVICE): # automatically manages which operations use which precision
                logits = model(images)
                probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_masks.append(masks.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_masks = np.concatenate(all_masks)

    # Per-class AUROC; skip a class if it has only one label value present otherwise it wont be able to calculate AUROC
    # AUROC evaluates how well the model's probability scores/ranking separate 
    # positive cases from negative cases across different decision thresholds.
    aurocs = {}
    for i, cls in enumerate(CLASS_COLUMNS):

        # Only use labels that are valid for this class
        valid = all_masks[:, i] == 1

        labels_i = all_labels[valid, i]
        probs_i = all_probs[valid, i]

        # AUROC requires both positive and negative samples
        if len(np.unique(labels_i)) < 2:
            continue

        aurocs[cls] = roc_auc_score(
            labels_i,
            probs_i
        )

    mean_auroc = (
        np.mean(list(aurocs.values()))
        if aurocs
        else float("nan")
    )

    return aurocs, mean_auroc

def main():
    train_ds = ChestXrayDataset(f"../combined_manifest_train.csv",
    f"../processed_data/train", f"../processed_data_chexpert/train", "train")

    nih_val_ds = ChestXrayDataset(
    "../nih_common_val.csv",
    "../processed_data/val",
    "../processed_data_chexpert/val",
    "val"
    )
    chexpert_val_ds = ChestXrayDataset(f"../chexpert_common_val.csv",
    f"../processed_data/val",
    f"../processed_data_chexpert/val",  "val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True) # 4 CPU workers prepare/load batch when CPU is working with another batch

    nih_val_loader = DataLoader(
    nih_val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

    chexpert_val_loader = DataLoader(
        chexpert_val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    model = build_model().to(DEVICE)

    class_weights = torch.tensor(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 1.0],
    device=DEVICE
    )

    criterion = FocalLoss(gamma=GAMMA, class_weights=class_weights)

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

        for step, (images, labels, masks) in enumerate(train_loader): # step indicates which batch currently on
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            masks = masks.to(DEVICE)

            with autocast(device_type=DEVICE):
                logits = model(images)
                loss = criterion(logits, labels, masks) / ACCUM_STEPS

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

        nih_aurocs, nih_mean = evaluate(model, nih_val_loader)
        chexpert_aurocs, chexpert_mean = evaluate(model, chexpert_val_loader)

        combined_score = (nih_mean + chexpert_mean) / 2

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - "
        f"LR: {optimizer.param_groups[0]['lr']:.2e} - "
        f"train loss: {avg_loss:.4f} - ")

        print(f" NIH mean AUROC:       {nih_mean:.4f}")
        for cls, score in nih_aurocs.items():
            print(f"   {cls:<18s} AUROC: {score:.4f}")

        print(f" CheXpert mean AUROC:  {chexpert_mean:.4f}")
        for cls, score in chexpert_aurocs.items():
            print(f"   {cls:<18s} AUROC: {score:.4f}")

        print(f" Combined mean AUROC:  {combined_score:.4f}")

        os.makedirs("checkpoints", exist_ok=True)

        epoch_ckpt_path = f"checkpoints/epoch_{epoch+1:02d}.pt"
        torch.save(model.state_dict(), epoch_ckpt_path)

        if combined_score > best_auroc:
            best_auroc = combined_score
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