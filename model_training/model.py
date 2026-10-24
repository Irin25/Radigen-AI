"""
model.py
Swin-Tiny backbone (timm, ImageNet-pretrained) with an 8-output
multi-label classification head.
"""
import timm
import torch.nn as nn

NUM_CLASSES = 8

def build_model(model_name="swin_tiny_patch4_window7_224", pretrained=True):
    model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=8,
    drop_path_rate=0.1, 
    drop_rate=0.1,        # dropout before the head
)
    # timm already swaps in a fresh Linear(..., NUM_CLASSES) head when
    # num_classes is passed, so no manual head replacement is needed.
    return model