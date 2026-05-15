"""Load PNG/JPEG or single-frame DICOM into RGB PIL Image."""

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None


def load_image_rgb(path: Path) -> Image.Image:
    """
    Load an image file as RGB uint8 PIL Image.
    Supports common raster formats and 2D DICOM (first frame).
    """
    suffix = path.suffix.lower()
    if suffix in {".dcm", ".dicom"}:
        if pydicom is None:
            raise RuntimeError("pydicom is required for DICOM support")
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        # Window to 0-255 (simple min-max if no window center/width)
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max() * 255.0
        arr = arr.astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    img = Image.open(path)
    return img.convert("RGB")


def bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")
