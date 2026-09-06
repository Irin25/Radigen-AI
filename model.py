"""
model.py
Swin-Tiny backbone (timm, ImageNet-pretrained) with an 8-output
multi-label classification head.
"""
import timm
import torch.nn as nn

NUM_CLASSES = 8

def build_model(model_name="swin_tiny_patch4_window7_224", pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=NUM_CLASSES)
    # timm already swaps in a fresh Linear(..., NUM_CLASSES) head when
    # num_classes is passed, so no manual head replacement is needed.
    return model
if __name__ == "__main__":
    import torch
    model = build_model()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape) # expect torch.Size([2, 8])