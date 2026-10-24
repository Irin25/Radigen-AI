import base64, os, threading
import cv2, numpy as np, torch
import torchvision.transforms as T
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import build_model
from dataset import CLASS_COLUMNS
from gradcam_explain import (get_target_layer, reshape_transform_swin,
                             load_thresholds, DEVICE, CKPT_PATH,
                             IMG_SIZE, VIS_SIZE, MEAN, STD, NO_FINDING)

# From threshold_values.txt: these classes did not reach the sens/spec target
BELOW_TARGET = {"Atelectasis", "Consolidation", "Pneumonia", "Pneumothorax"}

_tf = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])


class Predictor:
    def __init__(self):
        if not os.path.exists(CKPT_PATH):
            raise FileNotFoundError(
                f"Model file not found: {os.path.abspath(CKPT_PATH)}\n"
                "Put best_model.pt inside the backend/checkpoints/ folder."
            )
        if not os.path.exists("faiss_index/index.faiss"):
            raise FileNotFoundError(
                "faiss_index/index.faiss not found. Copy the faiss_index folder into backend/."
            )
        self.model = build_model(pretrained=False).to(DEVICE)
        self.model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
        self.model.eval()
        self.thresholds = load_thresholds()
        self.cam = GradCAMPlusPlus(
            model=self.model,
            target_layers=get_target_layer(self.model, -1),
            reshape_transform=reshape_transform_swin,
        )
        self.lock = threading.Lock()

    def _preprocess(self, stream):
        # Same recipe as the training data: grayscale -> 224x224 with INTER_AREA
        data = np.frombuffer(stream.read(), np.uint8)
        gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("That file isn't a readable image (use PNG/JPG)")
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        img = Image.fromarray(gray).convert("RGB")   # 1 channel -> 3, like training
        x = _tf(img).unsqueeze(0).to(DEVICE)
        rgb = np.stack([gray] * 3, axis=-1).astype(np.float32) / 255.0
        return x, rgb

    @staticmethod
    def _overlay_b64(rgb, cam_map):
        rgb_v = np.clip(cv2.resize(rgb, (VIS_SIZE, VIS_SIZE), interpolation=cv2.INTER_CUBIC), 0, 1)
        cam_v = np.clip(cv2.resize(cam_map, (VIS_SIZE, VIS_SIZE), interpolation=cv2.INTER_CUBIC), 0, 1)
        overlay = show_cam_on_image(rgb_v, cam_v, use_rgb=True)
        _, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return "data:image/png;base64," + base64.b64encode(buf).decode()

    def predict(self, stream):
        x, rgb = self._preprocess(stream)
        with self.lock:
            with torch.no_grad():
                probs = torch.sigmoid(self.model(x))[0].cpu().numpy()

            classes, findings = [], []
            for i, c in enumerate(CLASS_COLUMNS):
                t = float(self.thresholds[c])
                p = float(probs[i])
                classes.append({
                    "name": c, "probability": p, "threshold": t,
                    "positive": p >= t,
                    "reliability": "low" if c in BELOW_TARGET else "ok",
                })

            for item in classes:
                if item["positive"] and item["name"] != NO_FINDING:
                    idx = CLASS_COLUMNS.index(item["name"])
                    cam_map = self.cam(input_tensor=x,
                                       targets=[ClassifierOutputTarget(idx)])[0]
                    findings.append({**item, "heatmap": self._overlay_b64(rgb, cam_map)})

        return {"classes": classes, "findings": findings}
