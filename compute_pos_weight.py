"""
compute_pos_weight.py
Computes a per-class positive-weight tensor from manifest_train.csv,
for use in BCEWithLogitsLoss(pos_weight=...).
"""
import pandas as pd
import torch

CLASS_COLUMNS = [
"Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
"Pleural Effusion", "Pneumonia", "Pneumothorax", "No Finding",
]

def compute_pos_weight(manifest_csv):
    df = pd.read_csv(manifest_csv)
    pos_counts = df[CLASS_COLUMNS].sum()
    neg_counts = len(df) - pos_counts
    # Standard pos_weight formula: negatives / positives per class.
    # Classes with very few positives get a large weight; capped to 20 
    # to avoid extreme values from destabilizing early training.
    pos_weight = (neg_counts / pos_counts.clip(lower=1)).clip(upper=20.0) # no division by 0(in case of no positives) so clip 1
    print("Per-class pos_weight:")
    print(pos_weight)
    return torch.tensor(pos_weight.values, dtype=torch.float32)

if __name__ == "__main__":
    compute_pos_weight("../manifest_train.csv")