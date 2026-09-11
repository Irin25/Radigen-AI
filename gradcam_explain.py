"""
gradcam_explain.py
Generates a Grad-CAM heatmap overlay for a single chest X-ray,
using the classifier trained in train.py and per-class optimal thresholds
from compute_thresholds.py.

Usage:
    python gradcam_explain.py --image path/to/xray.png --class_name Pneumonia
    python gradcam_explain.py --image path/to/xray.png   # auto: highest-confidence class that clears its threshold
"""

import argparse
import json
import os
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms as T

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import build_model
from dataset import CLASS_COLUMNS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "checkpoints/best_model.pt"
THRESHOLDS_PATH = "thresholds.json"
IMG_SIZE = 224
DEFAULT_THRESHOLD = 0.5


def load_thresholds():
    if os.path.exists(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH, "r") as f:
            return json.load(f)
    print(f"Warning: {THRESHOLDS_PATH} not found, using flat 0.5 for all classes. "
          f"Run compute_thresholds.py first for calibrated results.")
    return {cls: DEFAULT_THRESHOLD for cls in CLASS_COLUMNS}


def load_model():
    model = build_model().to(DEVICE)
    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_target_layer(model):
    return [model.layers[-1].blocks[-1].norm1]


def reshape_transform_swin(tensor):
    return tensor.permute(0, 3, 1, 2)


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(img_resized).unsqueeze(0)
    rgb_img = np.array(img_resized).astype(np.float32) / 255.0
    return input_tensor, rgb_img


def pick_class(probs, thresholds, forced_class_name=None):
    """
    Returns (class_idx, class_name, raw_prob, cleared_threshold, normalized_confidence)

    normalized_confidence expresses how far above/below its OWN threshold
    the probability sits, on a 0-1 scale relative to that threshold's headroom.
    This is more meaningful than the raw sigmoid value across classes with
    very different optimal thresholds.
    """
    if forced_class_name is not None:
        idx = CLASS_COLUMNS.index(forced_class_name)
        t = thresholds.get(forced_class_name, DEFAULT_THRESHOLD)
        p = probs[idx]
        cleared = p >= t
        norm_conf = _normalized_confidence(p, t)
        return idx, forced_class_name, p, cleared, norm_conf

    # Auto mode: among classes that clear their OWN threshold, pick the one
    # with the highest margin above threshold. If none clear, fall back to
    # "No Finding" style result -- report the closest-to-threshold class.
    margins = []
    for i, cls in enumerate(CLASS_COLUMNS):
        t = thresholds.get(cls, DEFAULT_THRESHOLD)
        margins.append(probs[i] - t)
    margins = np.array(margins)

    cleared_mask = margins >= 0
    if cleared_mask.any():
        idx = int(np.argmax(np.where(cleared_mask, margins, -np.inf)))
    else:
        idx = int(np.argmax(margins))  # closest to clearing, even if none did

    cls = CLASS_COLUMNS[idx]
    t = thresholds.get(cls, DEFAULT_THRESHOLD)
    p = probs[idx]
    cleared = p >= t
    norm_conf = _normalized_confidence(p, t)
    return idx, cls, p, cleared, norm_conf


def _normalized_confidence(p, t):
    """
    Maps p relative to t onto 0-1:
      p == t        -> 0.5
      p == 1.0      -> 1.0  (if p > t)
      p == 0.0      -> 0.0  (if p < t)
    Gives an interpretable "how confident, relative to this class's own
    calibrated cutoff" number instead of comparing raw sigmoid outputs
    across classes with very different thresholds.
    """
    if p >= t:
        if t >= 1.0:
            return 1.0
        return 0.5 + 0.5 * (p - t) / (1.0 - t)
    else:
        if t <= 0.0:
            return 0.0
        return 0.5 * (p / t)


def run_gradcam(image_path, class_name=None, output_path="gradcam_output.png"):
    thresholds = load_thresholds()
    model = load_model()
    input_tensor, rgb_img = preprocess_image(image_path)
    input_tensor = input_tensor.to(DEVICE)
    target_layers = get_target_layer(model)

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform_swin,
    )

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    class_idx, picked_class, raw_prob, cleared, norm_conf = pick_class(
        probs, thresholds, forced_class_name=class_name
    )

    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    status = "POSITIVE" if cleared else "below threshold (not called positive)"
    print(f"\nPredicted focus class: {picked_class}")
    print(f"  Raw probability:        {raw_prob:.3f}")
    print(f"  Class threshold:        {thresholds.get(picked_class, DEFAULT_THRESHOLD):.3f}")
    print(f"  Status:                 {status}")
    print(f"  Normalized confidence:  {norm_conf:.3f}  (0.5 = right at threshold)")
    print(f"Heatmap saved to: {output_path}")

    # Optional: print full breakdown across all 8 classes
    print("\nAll classes:")
    for i, cls in enumerate(CLASS_COLUMNS):
        t = thresholds.get(cls, DEFAULT_THRESHOLD)
        mark = "  <-- positive" if probs[i] >= t else ""
        print(f"  {cls:<18s} prob={probs[i]:.3f}  thresh={t:.3f}{mark}")

    return output_path, picked_class, raw_prob, cleared, norm_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to chest X-ray image")
    parser.add_argument("--class_name", default=None,
                         help="One of the 8 class names. Omit to auto-pick highest-margin class.")
    parser.add_argument("--output", default="gradcam_output.png")
    args = parser.parse_args()

    run_gradcam(args.image, args.class_name, args.output)