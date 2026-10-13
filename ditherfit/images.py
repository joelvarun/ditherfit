"""Test image handling: fetch, cache, decode and sample.

The benchmark runs on the Kodak PhotoCD set, the standard reference images for
colour quantization work. They are downloaded on first use and cached under
data/, which is not committed.
"""

import os
import urllib.request

import numpy as np
from PIL import Image

from .color import lightness_weight, srgb_to_linear
from .metrics import noise_visibility

KODAK_URL = "https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
CALIBRATION = (5, 13)
EVALUATION = (23, 7, 15, 19)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def fetch_kodak(number, data_dir=DATA_DIR):
    """Return a local path to Kodak image `number`, downloading if needed."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "kodim{:02d}.png".format(number))
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        url = KODAK_URL.format(number)
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        with open(path, "wb") as handle:
            handle.write(payload)
    return path


def open_resized(path, long_side=384):
    """Open an image and resize it so the long side is `long_side` pixels."""
    image = Image.open(path).convert("RGB")
    w, h = image.size
    scale = long_side / float(max(w, h))
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return image.resize(size, Image.LANCZOS)


def to_linear(image):
    """PIL RGB image to a linear light float array."""
    return srgb_to_linear(np.asarray(image, dtype=np.float64) / 255.0)


def sample_pixels(lin, count=20000, seed=0):
    """Sample pixels with their accuracy weight and dither visibility weight."""
    lin = np.asarray(lin, dtype=np.float64)
    height, width, _ = lin.shape
    weights = lightness_weight(lin).ravel()
    visibility = noise_visibility(lin).ravel()
    total = height * width
    rng = np.random.default_rng(seed)
    take = min(count, total)
    sel = rng.choice(total, size=take, replace=False)
    return lin.reshape(-1, 3)[sel], weights[sel], visibility[sel]
