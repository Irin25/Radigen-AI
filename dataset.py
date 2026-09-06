"""
dataset.py
PyTorch Dataset for the RadiGen AI multi-label chest X-ray classifier.
Reads image filenames + 8 multi-hot label columns from a manifest CSV,
loads the corresponding PNG from processed_data/<split>/, and returns
(image_tensor, label_tensor).
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

CLASS_COLUMNS = [
"Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
"Pleural Effusion", "Pneumonia", "Pneumothorax", "No Finding",
]

# ImageNet normalization stats, since Swin-Tiny is ImageNet-pretrained
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def to_three_channels(x):
    return x.repeat(3, 1, 1)

def build_transforms(split):

    if split == "train":
        # Light augmentation for training
        return T.Compose([
        T.RandomRotation(degrees=5),
        T.ColorJitter(brightness=0.1, contrast=0.1), # change brightness and contrast by up to 10%
        T.ToTensor(), # scales pixel values to 0-1
        T.Lambda(to_three_channels), # 1 channel -> 3 channels
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), # normalizes the scaled values
        ])

    # For validation/testing data, it skips rotations and brightness changes to evaluate the model fairly on real, unmodified images.
    else:
        return T.Compose([
        T.ToTensor(),
        T.Lambda(to_three_channels),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    
class ChestXrayDataset(Dataset):

    def __init__(self, manifest_csv, image_dir, split):
        df_raw = pd.read_csv(manifest_csv)
        self.image_dir = image_dir
        self.transform = build_transforms(split)

        # Filter out rows whose image files don't physically exist on disk
        valid_indices = []
        for idx in range(len(df_raw)):
            filename = os.path.basename(df_raw.iloc[idx]["image_filename"])
            img_path = os.path.join(self.image_dir, filename)
            if os.path.exists(img_path):
                valid_indices.append(idx)

        # Keep only existing images and reset the index
        self.df = df_raw.iloc[valid_indices].reset_index(drop=True)
        print(f"Loaded {len(self.df)}/{len(df_raw)} images for split: {split}")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = os.path.basename(row["image_filename"])
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("L") # grayscale
        image = self.transform(image)
        labels = torch.tensor(row[CLASS_COLUMNS].values.astype("float32"))
        return image, labels

