import os
import math
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
NUM_EPOCHS = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FREEZE_EPOCHS = 1
BACKBONE_LR = 2e-5
HEAD_LR = 2e-4

PATIENCE = 3

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits, targets, masks):
        logits = logits.float()
        targets = targets.float()
        masks = masks.float()

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        w = targets * self.pos_weight + (1 - targets)   # pos_weight on positives, 1 on negatives
        loss = bce * w * masks                          # ignore masked-out labels
        return loss.sum() / masks.sum().clamp(min=1.0)

def evaluate(model, loader):
    model.eval()
    all_probs, all_labels, all_masks = [], [], []
    with torch.no_grad(): # no gradient descent done during evaluation , just finding its accuracy before training
        for images, labels, masks in loader:
            images = images.to(DEVICE)
            with autocast(device_type=DEVICE): # automatically manages which operations use which precision
                logits = model(images)
                probs = torch.sigmoid(logits.float())
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

    # Get the classification head from the timm model
    head = model.get_classifier()
    head_param_ids = {id(p) for p in head.parameters()}

    # Freeze the entire model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the classification head
    for param in head.parameters():
        param.requires_grad = True

    # Separate backbone and head parameters
    backbone_params = [
        p for p in model.parameters()
        if id(p) not in head_param_ids
    ]

    head_params = list(head.parameters())

    # Include both groups in the optimizer from the start.
    # Frozen backbone parameters will not receive gradients yet.
    optimizer = torch.optim.AdamW( 
    [{"params": backbone_params, "lr": BACKBONE_LR},   # param_groups[0]
    {"params": head_params,     "lr": HEAD_LR}],       # param_groups[1],
        weight_decay=0.05
    )

    pos_w = torch.load("../combined_pos_weight.pt").float()
    pos_w = pos_w.sqrt().clamp(max=5.0)
    pos_w[CLASS_COLUMNS.index("No Finding")] = 1.0
    pos_w = pos_w.to(DEVICE)
    
    criterion = WeightedBCELoss(pos_w).to(DEVICE)

    steps_per_epoch = len(train_loader) # number of batches in one epoch
    total_steps = NUM_EPOCHS * steps_per_epoch
    warmup_steps = steps_per_epoch            # 1 epoch of warmup
    freeze_steps = FREEZE_EPOCHS * steps_per_epoch

    def cosine(progress):
        return 0.5 * (1 + math.cos(math.pi * progress)) # Gradually decreases the learning rate from its full value to zero.

    def backbone_lambda(step): # # Keeps the backbone frozen initially, then warms up its learning rate and gradually decreases it.
        if step < freeze_steps:
            return 1.0                        # frozen, value doesn't matter
        s = step - freeze_steps
        if s < warmup_steps:
            return (s + 1) / warmup_steps     # warmup starts at unfreeze
        return cosine((s - warmup_steps) / max(1, total_steps - freeze_steps - warmup_steps))

    def head_lambda(step):  # Gradually warms up the classification head's learning rate during the first epoch, then decreases it using cosine decay.
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return cosine((step - warmup_steps) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=[backbone_lambda, head_lambda]   # same order as the groups
    )

    scaler = GradScaler(device=DEVICE) # helps prevent very small FP16 gradients from becoming numerically useless during training.
    best_auroc = 0.0
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):

         # Unfreeze the backbone after the initial frozen epochs
        if epoch == FREEZE_EPOCHS:
            for param in model.parameters():
                param.requires_grad = True

            print(
                "\nBackbone unfrozen! "
                "Full-model fine-tuning begins."
            )

        elif epoch < FREEZE_EPOCHS:
            for p in backbone_params:
                p.requires_grad = False

        trainable = sum(p.requires_grad for p in model.parameters())
        print(f"Epoch {epoch+1}: {trainable} trainable parameter tensors")

        model.train()

        running_loss = 0.0
        optimizer.zero_grad()

        for step,(images, labels, masks) in enumerate(train_loader): 
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            masks = masks.to(DEVICE)

            with autocast(device_type=DEVICE):
                logits = model(images)
                loss = criterion(logits, labels, masks) 

            scaler.scale(loss).backward()
          
            scaler.step(optimizer) 
            scaler.update()
            optimizer.zero_grad()
            scheduler.step() 
            running_loss += loss.item() 

        avg_loss = running_loss / len(train_loader)

        nih_aurocs, nih_mean = evaluate(model, nih_val_loader)
        chexpert_aurocs, chexpert_mean = evaluate(model, chexpert_val_loader)

        combined_score = (nih_mean + chexpert_mean) / 2

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - "
        f"LR backbone: {optimizer.param_groups[0]['lr']:.2e} - "
        f"LR head: {optimizer.param_groups[1]['lr']:.2e} - "
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
            epochs_no_improve = 0
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            print(f" -> New best model saved (mean AUROC {best_auroc:.4f})")
        elif epoch >= FREEZE_EPOCHS:          # don't count the frozen phase
            epochs_no_improve += 1
            print(f" No improvement for {epochs_no_improve}/{PATIENCE} epochs")
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break
       

    print(f"\nTraining complete. Best validation mean AUROC: {best_auroc:.4f}")

if __name__ == "__main__":
    main()