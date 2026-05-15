"""
Grad-CAM for CNN explainability (Selvaraju et al., 2017).
Works with ResNet-style layers: target_layer should produce spatial feature maps.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Compute a coarse localization map highlighting important regions."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, device: torch.device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _save_activation(self, _module, _inp, out):
        self.activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def register(self) -> None:
        self._handles.append(self.target_layer.register_forward_hook(self._save_activation))
        self._handles.append(self.target_layer.register_full_backward_hook(self._save_gradient))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.no_grad()
    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> np.ndarray:
        """
        Forward + backward for one sample. Returns HxW heatmap in [0,1].
        If class_idx is None, uses argmax of logits.
        """
        self.model.eval()
        self.gradients = None
        self.activations = None

        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx].sum()
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # global average pooling of gradients
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.astype(np.float32)


def overlay_heatmap_on_image(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Resize heatmap to image size and blend as a red overlay."""
    import cv2

    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hm_uint8 = (hm * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    blended = (alpha * heat_color + (1 - alpha) * img).astype(np.uint8)
    return Image.fromarray(blended)


def default_resnet_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Return layer4 for torchvision ResNet."""
    if not hasattr(model, "layer4"):
        raise ValueError("Model has no layer4; provide a custom target_layer for Grad-CAM")
    return model.layer4
