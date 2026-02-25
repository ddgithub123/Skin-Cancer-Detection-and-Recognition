"""
model_manager.py
─────────────────
Single source of truth for model metadata and lazy loading.

Each model entry declares:
  - path        : .h5 file location
  - backbone    : architecture name (used to select preprocessing + Grad-CAM layer)
  - num_classes : 1 (binary/sigmoid) or N (multiclass/softmax)
  - labels      : class index → human label mapping
  - last_conv_hint : optional known layer name; set to None to auto-detect

The manager loads models lazily (on first use) and caches them in memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

# ──────────────────────────────────────────────
# Model Registry
# ──────────────────────────────────────────────

HAM10000_LABELS: Dict[int, str] = {
    0: "Actinic keratoses (akiec)",
    1: "Basal cell carcinoma (bcc)",
    2: "Benign keratosis-like lesions (bkl)",
    3: "Dermatofibroma (df)",
    4: "Melanoma (mel)",
    5: "Melanocytic nevi (nv)",
    6: "Vascular lesions (vasc)",
}

BINARY_LABELS: Dict[int, str] = {
    0: "Non-Melanoma",
    1: "Melanoma",
}


@dataclass
class ModelConfig:
    """Declarative configuration for a single model."""
    task: str                        # "binary" | "multiclass"
    path: str                        # absolute or relative path to .h5
    backbone: str                    # "efficientnetb0" | "resnet50" | "efficientnetb3" …
    num_classes: int                 # 1 for sigmoid, N for softmax
    labels: Dict[int, str]
    last_conv_hint: Optional[str] = None   # override auto-detection when known


# ── Registry ── edit here to add / change models ──────────────────────────────

_BASE = os.path.join(os.path.dirname(__file__))

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "binary": ModelConfig(
        task="binary",
        path=os.path.join(_BASE, "binary_model_converted.keras"),
        backbone="efficientnetb0",
        num_classes=1,
        labels=BINARY_LABELS,
        # EfficientNetB0 last conv block output (auto-detect will find this anyway)
        last_conv_hint=None,
    ),
    "multiclass": ModelConfig(
        task="multiclass",
        path=os.path.join(_BASE, "multiclass_model.h5"),
        backbone="ResNet50",   # ← fix: was ResNet in old code; set to match actual .h5
        num_classes=7,
        labels=HAM10000_LABELS,
        last_conv_hint=None,
    ),
}

# ──────────────────────────────────────────────
# Lazy loader
# ──────────────────────────────────────────────

_model_cache: Dict[str, object] = {}
_HAS_TF: Optional[bool] = None


def _check_tf() -> bool:
    global _HAS_TF
    if _HAS_TF is None:
        try:
            import tensorflow  # noqa: F401
            _HAS_TF = True
        except ImportError:
            _HAS_TF = False
    return _HAS_TF


def get_model(task: str):
    """Return a loaded Keras model for *task*, or None if unavailable."""

    if not _check_tf():
        return None

    if task in _model_cache:
        return _model_cache[task]

    cfg = MODEL_REGISTRY.get(task)
    if cfg is None:
        raise ValueError(f"Unknown task: {task!r}. Choose from {list(MODEL_REGISTRY)}")

    if not os.path.exists(cfg.path):
        return None

    import tensorflow as tf

    print(f"Loading model for task: {task}")
    print("Path:", cfg.path)

    # Load both .keras and .h5 safely
    model = tf.keras.models.load_model(cfg.path, compile=False)

    # Warm-up forward pass (important for Grad-CAM)
    try:
        input_shape = model.input_shape
        dummy = tf.zeros((1,) + tuple(input_shape[1:]))
        _ = model(dummy, training=False)
        print("Model warm-up complete ✔")
    except Exception as e:
        print("Warm-up skipped:", e)

    _model_cache[task] = model
    return model


def get_config(task: str) -> ModelConfig:
    cfg = MODEL_REGISTRY.get(task)
    if cfg is None:
        raise ValueError(f"Unknown task: {task!r}")
    return cfg


def has_tf() -> bool:
    return _check_tf()


def model_available(task: str) -> bool:
    cfg = MODEL_REGISTRY.get(task)
    return cfg is not None and os.path.exists(cfg.path) and _check_tf()