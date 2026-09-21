"""
gradcam++_explain.py
Generates a Grad-CAM++ heatmap overlay for a single chest X-ray,
using the Swin-T classifier trained in train.py and per-class optimal
thresholds from compute_thresholds.py.

Usage:
    python gradcam++_explain.py --image path/to/xray.png --class_name Pneumonia
    python gradcam++_explain.py --image path/to/xray.png                # auto-pick class
    python gradcam++_explain.py --image path/to/xray.png --stage -2     # finer 14x14 map
    python gradcam++_explain.py --image path/to/xray.png --all_positive # one panel per positive class
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import build_model
from dataset import CLASS_COLUMNS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "checkpoints/best_model.pt"
THRESHOLDS_PATH = "thresholds.json"
IMG_SIZE = 224          # must match training
VIS_SIZE = 448          # output visualization size (display only)
DEFAULT_THRESHOLD = 0.5
NO_FINDING = "No Finding"

# IMPORTANT: keep identical to the eval transform in dataset.py
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_thresholds():
    if os.path.exists(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH, "r") as f:
            return json.load(f)
    print(f"Warning: {THRESHOLDS_PATH} not found, using flat {DEFAULT_THRESHOLD} "
          f"for all classes. Run compute_thresholds.py for calibrated results.")
    return {cls: DEFAULT_THRESHOLD for cls in CLASS_COLUMNS}


def load_model():
    model = build_model().to(DEVICE)
    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
    input_tensor = transform(img_resized).unsqueeze(0)
    rgb_img = np.array(img_resized).astype(np.float32) / 255.0
    return input_tensor, rgb_img


# --------------------------------------------------------------------------
# Swin-specific CAM plumbing
# --------------------------------------------------------------------------
def get_target_layer(model, stage=-1):
    """
    stage -1: last stage, 7x7 grid (coarse)
    stage -2: second-to-last stage, 14x14 grid (finer localization)
    """
    return [model.layers[stage].blocks[-1].norm1]


def reshape_transform_swin(tensor):
    """Convert Swin token activations to (B, C, H, W). Handles both timm layouts."""
    if tensor.dim() == 4:                       # (B, H, W, C) -- timm >= 0.9
        return tensor.permute(0, 3, 1, 2)
    b, l, c = tensor.shape                      # (B, H*W, C) -- older timm
    hw = int(round(l ** 0.5))
    return tensor.reshape(b, hw, hw, c).permute(0, 3, 1, 2)


# --------------------------------------------------------------------------
# Class selection
# --------------------------------------------------------------------------
def _normalized_confidence(p, t):
    """
    Position of p relative to its own threshold t, on a 0-1 scale:
      p == t -> 0.5,  p == 1 -> 1.0,  p == 0 -> 0.0
    """
    if p >= t:
        return 1.0 if t >= 1.0 else 0.5 + 0.5 * (p - t) / (1.0 - t)
    return 0.0 if t <= 0.0 else 0.5 * (p / t)


def _info(probs, thresholds, idx):
    cls = CLASS_COLUMNS[idx]
    t = thresholds.get(cls, DEFAULT_THRESHOLD)
    p = float(probs[idx])
    return idx, cls, p, p >= t, _normalized_confidence(p, t)


def pick_class(probs, thresholds, forced_class_name=None):
    """
    Returns (class_idx, class_name, raw_prob, cleared_threshold, normalized_confidence).
    Auto mode prefers pathology classes over 'No Finding' (a heatmap for 'no
    disease' is not meaningful).
    """
    if forced_class_name is not None:
        return _info(probs, thresholds, CLASS_COLUMNS.index(forced_class_name))
 
    margins = np.array([probs[i] - thresholds.get(c, DEFAULT_THRESHOLD)
                        for i, c in enumerate(CLASS_COLUMNS)])
    is_path = np.array([c != NO_FINDING for c in CLASS_COLUMNS])
    cleared = margins >= 0
 
    if (cleared & is_path).any():
        idx = int(np.argmax(np.where(cleared & is_path, margins, -np.inf)))
    elif cleared.any():
        idx = int(np.argmax(np.where(cleared, margins, -np.inf)))
    else:
        idx = int(np.argmax(margins))
    return _info(probs, thresholds, idx)
# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------
def make_panel(rgb_img, cam_map, title):
    """Side-by-side: original | Grad-CAM++ overlay, with a title bar."""
    rgb_vis = np.clip(cv2.resize(rgb_img, (VIS_SIZE, VIS_SIZE),
                                 interpolation=cv2.INTER_CUBIC), 0, 1)
    cam_vis = np.clip(cv2.resize(cam_map, (VIS_SIZE, VIS_SIZE),
                                 interpolation=cv2.INTER_CUBIC), 0, 1)
    overlay = show_cam_on_image(rgb_vis, cam_vis, use_rgb=True)
    original = (rgb_vis * 255).astype(np.uint8)

    panel = np.concatenate([original, overlay], axis=1)
    bar = np.full((36, panel.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(bar, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([bar, panel], axis=0)


def compute_cam(cam, input_tensor, class_idx):
    return cam(input_tensor=input_tensor,
               targets=[ClassifierOutputTarget(class_idx)])[0, :]


# --------------------------------------------------------------------------
# Main routine
# --------------------------------------------------------------------------
def run_gradcam(image_path, class_name=None, output_path="gradcam_output.png",
                stage=-1, all_positive=False):
    thresholds = load_thresholds()
    model = load_model()
    input_tensor, rgb_img = preprocess_image(image_path)
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        probs = torch.sigmoid(model(input_tensor)).cpu().numpy()[0]

    cam = GradCAMPlusPlus(
        model=model,
        target_layers=get_target_layer(model, stage),
        reshape_transform=reshape_transform_swin,
    )

    # Decide which classes to explain
    if all_positive and class_name is None:
        idxs = [i for i, c in enumerate(CLASS_COLUMNS)
                if c != NO_FINDING and probs[i] >= thresholds.get(c, DEFAULT_THRESHOLD)]
        if not idxs:   # nothing positive -> fall back to auto pick
            idxs = [pick_class(probs, thresholds)[0]]
    else:
        idxs = [pick_class(probs, thresholds, class_name)[0]]

    panels = []
    results = []
    for idx in idxs:
        _, cls, p, cleared, norm_conf = _info(probs, thresholds, idx)
        cam_map = compute_cam(cam, input_tensor, idx)
        title = (f"{cls}  p={p:.3f}  thr={thresholds.get(cls, DEFAULT_THRESHOLD):.3f}  "
                 f"{'POSITIVE' if cleared else 'below thr'}")
        panels.append(make_panel(rgb_img, cam_map, title))
        results.append((cls, p, cleared, norm_conf))

    final = np.concatenate(panels, axis=0)
    cv2.imwrite(output_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

    # ---- console report ----
    for cls, p, cleared, norm_conf in results:
        status = "POSITIVE" if cleared else "below threshold (not called positive)"
        print(f"\nExplained class: {cls}")
        print(f"  Raw probability:        {p:.3f}")
        print(f"  Class threshold:        {thresholds.get(cls, DEFAULT_THRESHOLD):.3f}")
        print(f"  Status:                 {status}")
        print(f"  Normalized confidence:  {norm_conf:.3f}  (0.5 = right at threshold)")
        if not cleared:
            print("  Note: class not predicted positive, so this map is not meaningful evidence.")
    print(f"\nHeatmap saved to: {output_path}  (stage={stage}, "
          f"grid={'7x7' if stage == -1 else '14x14'})")

    print("\nAll classes:")
    for i, c in enumerate(CLASS_COLUMNS):
        t = thresholds.get(c, DEFAULT_THRESHOLD)
        mark = "  <-- positive" if probs[i] >= t else ""
        print(f"  {c:<18s} prob={probs[i]:.3f}  thresh={t:.3f}{mark}")

    return output_path, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to chest X-ray image")
    parser.add_argument("--class_name", default=None, choices=CLASS_COLUMNS,
                        help="Class to explain. Omit to auto-pick the highest-margin pathology.")
    parser.add_argument("--output", default="gradcam_output.png")
    parser.add_argument("--stage", type=int, default=-1, choices=[-1, -2],
                        help="-1: 7x7 map (coarse), -2: 14x14 map (finer)")
    parser.add_argument("--all_positive", action="store_true",
                        help="Produce one panel for every pathology predicted positive")
    args = parser.parse_args()

    run_gradcam(args.image, args.class_name, args.output, args.stage, args.all_positive)