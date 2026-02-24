"""
preprocessing.py
────────────────
Backbone-specific image preprocessing.

Each backbone trained with different normalization:
  - EfficientNet (all variants) : tf.keras.applications.efficientnet.preprocess_input
    → scales to [-1, 1] (not plain /255 as in the original code — this was a bug)
  - ResNet50 / ResNet101        : tf.keras.applications.resnet.preprocess_input
    → zero-centers per ImageNet channel means (BGR)
  - VGG16 / VGG19              : tf.keras.applications.vgg16.preprocess_input
  - InceptionV3                 : tf.keras.applications.inception_v3.preprocess_input

The public API is:
    preprocess(image_path_or_array, backbone) -> np.ndarray shape (1, H, W, 3)

PIL.Image or file-path inputs are both accepted.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from typing import Union

IMG_SIZE = (224, 224)

# ──────────────────────────────────────────────
# Backbone → preprocessing function mapping
# ──────────────────────────────────────────────

def _get_preprocess_fn(backbone: str):
    """Return the correct Keras preprocessing function for a backbone."""
    backbone = backbone.lower()

    try:
        import tensorflow as tf

        if backbone.startswith("efficientnet"):
            return tf.keras.applications.efficientnet.preprocess_input

        if backbone.startswith("resnet"):
            return tf.keras.applications.resnet.preprocess_input

        if backbone.startswith("vgg"):
            return tf.keras.applications.vgg16.preprocess_input

        if backbone.startswith("inception"):
            return tf.keras.applications.inception_v3.preprocess_input

        if backbone.startswith("mobilenet"):
            return tf.keras.applications.mobilenet_v2.preprocess_input

        # Fallback: plain /255 normalisation
        return lambda x: x / 255.0

    except ImportError:
        # No TF: return a simple normaliser
        return lambda x: x / 255.0


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def load_image(source: Union[str, "np.ndarray", Image.Image]) -> np.ndarray:
    """
    Accept a file path, PIL Image, or uint8 numpy array.
    Returns a uint8 numpy array of shape (224, 224, 3).
    """
    if isinstance(source, str):
        img = Image.open(source).convert("RGB").resize(IMG_SIZE)
        return np.array(img, dtype=np.uint8)

    if isinstance(source, Image.Image):
        img = source.convert("RGB").resize(IMG_SIZE)
        return np.array(img, dtype=np.uint8)

    # Assume numpy array
    arr = np.array(source)
    if arr.ndim == 3 and arr.shape[2] == 4:          # RGBA → RGB
        arr = arr[:, :, :3]
    if arr.shape[:2] != IMG_SIZE:
        img = Image.fromarray(arr.astype(np.uint8)).resize(IMG_SIZE)
        arr = np.array(img)
    return arr.astype(np.uint8)


def preprocess(
    source: Union[str, "np.ndarray", Image.Image],
    backbone: str,
) -> "np.ndarray":
    """
    Load + backbone-specific preprocessing.

    Returns:
        np.ndarray of shape (1, 224, 224, 3) ready for model.predict()
    """
    img_uint8 = load_image(source)                           # (H, W, 3) uint8
    img_float = img_uint8.astype("float32")
    img_batch = np.expand_dims(img_float, axis=0)            # (1, H, W, 3)

    fn = _get_preprocess_fn(backbone)
    return fn(img_batch)                                     # (1, H, W, 3) float32


def preprocess_display(
    source: Union[str, "np.ndarray", Image.Image],
) -> np.ndarray:
    """
    Returns a plain uint8 RGB array suitable for display (no normalization).
    """
    return load_image(source)