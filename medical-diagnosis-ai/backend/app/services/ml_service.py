"""
Model loading, inference, Grad-CAM, and lightweight segmentation.
Checkpoints are optional: if missing, ImageNet pretrained heads still produce plausible demo outputs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.config import get_settings
from app.ml.architectures import (
    BRAIN_LABELS,
    CHEST_LABELS,
    BrainMRINet,
    MultiLabelChestNet,
    UNet,
)
from app.ml.preprocess import brain_transforms_infer, chest_transforms_infer
from app.utils.gradcam import GradCAM, overlay_heatmap_on_image
from app.utils.image_io import load_image_rgb

logger = logging.getLogger(__name__)
settings = get_settings()


class MLService:
    """Singleton-style service: loads models once per process."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._chest: MultiLabelChestNet | None = None
        self._brain: BrainMRINet | None = None
        self._unet: UNet | None = None
        self._chest_tf = chest_transforms_infer()
        self._brain_tf = brain_transforms_infer()

    def _checkpoint_dir(self) -> Path:
        p = Path(settings.ml_checkpoints_dir)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[2] / p
        return p

    def load_models(self) -> None:
        """Load or initialize networks and optional state_dicts."""
        ckpt = self._checkpoint_dir()
        ckpt.mkdir(parents=True, exist_ok=True)

        self._chest = MultiLabelChestNet(num_labels=len(CHEST_LABELS), backbone="resnet50")
        chest_path = ckpt / "chest_resnet50.pt"
        if chest_path.exists():
            self._chest.load_state_dict(torch.load(chest_path, map_location="cpu", weights_only=True))
            logger.info("Loaded chest checkpoint from %s", chest_path)
        else:
            logger.warning("No chest checkpoint at %s — using ImageNet head (demo mode)", chest_path)
        self._chest.to(self.device)
        self._chest.eval()

        self._brain = BrainMRINet(num_classes=len(BRAIN_LABELS))
        brain_path = ckpt / "brain_resnet18.pt"
        if brain_path.exists():
            self._brain.load_state_dict(torch.load(brain_path, map_location="cpu", weights_only=True))
            logger.info("Loaded brain MRI checkpoint from %s", brain_path)
        else:
            logger.warning("No brain checkpoint at %s — using ImageNet head (demo mode)", brain_path)
        self._brain.to(self.device)
        self._brain.eval()

        self._unet = UNet(in_ch=3, base=32)
        unet_path = ckpt / "unet_segmentation.pt"
        if unet_path.exists():
            self._unet.load_state_dict(torch.load(unet_path, map_location="cpu", weights_only=True))
            logger.info("Loaded U-Net checkpoint from %s", unet_path)
        else:
            logger.warning("No U-Net checkpoint — segmentation will use saliency fallback")
        self._unet.to(self.device)
        self._unet.eval()

    def ensure_loaded(self) -> None:
        if self._chest is None:
            self.load_models()

    @staticmethod
    def _sigmoid_np(logits: torch.Tensor) -> dict[str, float]:
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        return {CHEST_LABELS[i]: float(probs[i]) for i in range(len(CHEST_LABELS))}

    @staticmethod
    def _softmax_np(logits: torch.Tensor) -> dict[str, float]:
        probs = F.softmax(logits, dim=1).detach().cpu().numpy().reshape(-1)
        return {BRAIN_LABELS[i]: float(probs[i]) for i in range(len(BRAIN_LABELS))}

    def predict_chest(self, image_path: Path) -> dict[str, Any]:
        self.ensure_loaded()
        assert self._chest is not None
        pil = load_image_rgb(image_path)
        tensor = self._chest_tf(pil).unsqueeze(0)
        with torch.no_grad():
            logits = self._chest(tensor.to(self.device))
        labels = self._sigmoid_np(logits)
        top = sorted(labels.items(), key=lambda x: x[1], reverse=True)[:3]
        top_findings = [f"{k} ({v:.2f})" for k, v in top]
        severity = float(np.clip(np.max(list(labels.values())), 0.0, 1.0))
        return {"labels": labels, "top_findings": top_findings, "severity_score": severity, "pil": pil, "tensor": tensor}

    def predict_brain(self, image_path: Path) -> dict[str, Any]:
        self.ensure_loaded()
        assert self._brain is not None
        pil = load_image_rgb(image_path)
        tensor = self._brain_tf(pil).unsqueeze(0)
        with torch.no_grad():
            logits = self._brain(tensor.to(self.device))
        labels = self._softmax_np(logits)
        pred_idx = int(logits.argmax(dim=1).item())
        top_findings = [f"{BRAIN_LABELS[pred_idx]} ({labels[BRAIN_LABELS[pred_idx]]:.2f})"]
        severity = 0.0 if pred_idx == BRAIN_LABELS.index("normal") else float(labels[BRAIN_LABELS[pred_idx]])
        return {"labels": labels, "top_findings": top_findings, "severity_score": severity, "pil": pil, "tensor": tensor}

    def gradcam_chest(self, tensor: torch.Tensor, pil: Image.Image, out_path: Path) -> str | None:
        self.ensure_loaded()
        assert self._chest is not None
        try:
            model, target = self._chest.gradcam_submodel()
        except Exception as e:  # pragma: no cover
            logger.error("Grad-CAM setup failed: %s", e)
            return None
        with torch.no_grad():
            idx = int(self._chest(tensor.to(self.device)).argmax(dim=1).item())
        cam = GradCAM(model, target, self.device)
        cam.register()
        try:
            heatmap = cam.generate_heatmap(tensor, class_idx=idx)
        finally:
            cam.remove()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        overlay = overlay_heatmap_on_image(pil, heatmap)
        overlay.save(out_path)
        return str(out_path)

    def gradcam_brain(self, tensor: torch.Tensor, pil: Image.Image, out_path: Path) -> str | None:
        self.ensure_loaded()
        assert self._brain is not None
        model, target = self._brain.gradcam_submodel()
        cam = GradCAM(model, target, self.device)
        cam.register()
        try:
            heatmap = cam.generate_heatmap(tensor, class_idx=None)
        finally:
            cam.remove()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        overlay = overlay_heatmap_on_image(pil, heatmap)
        overlay.save(out_path)
        return str(out_path)

    def segment(self, pil: Image.Image, modality: str, out_path: Path) -> str | None:
        """Run U-Net if weights exist; otherwise threshold on grayscale edge map (demo overlay)."""
        self.ensure_loaded()
        assert self._unet is not None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = self._checkpoint_dir() / "unet_segmentation.pt"
        arr = np.array(pil.resize((256, 256)).convert("RGB"), dtype=np.float32) / 255.0
        chw = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        chw = (chw - 0.45) / 0.25  # rough norm for unet demo if trained with similar

        if ckpt.exists():
            with torch.no_grad():
                logits = self._unet(chw)
                mask = torch.sigmoid(logits)[0, 0].cpu().numpy()
        else:
            gray = chw.mean(dim=1, keepdim=True)
            gx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
            gy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
            pad_gx = F.pad(gx, (0, 1, 0, 0))
            pad_gy = F.pad(gy, (0, 0, 0, 1))
            sal = (pad_gx + pad_gy).squeeze().cpu().numpy()
            mask = (sal > np.percentile(sal, 92)).astype(np.float32)

        import cv2

        h, w = np.array(pil).shape[:2]
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        base = np.array(pil.convert("RGB"))
        overlay = base.copy()
        color = np.zeros_like(base)
        color[:, :, 1] = (mask_resized * 255).astype(np.uint8)
        blended = cv2.addWeighted(base, 0.75, color, 0.45, 0)
        Image.fromarray(blended).save(out_path)
        return str(out_path)


ml_service = MLService()


def labels_to_json(d: dict[str, float]) -> str:
    return json.dumps(d)
