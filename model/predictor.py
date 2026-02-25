"""
predictor.py
────────────
Pure inference logic. Depends on model_manager and preprocessing.
No model loading, no visualization, no session state.

Public functions:
    predict(task, source)          -> PredictionResult
    get_feature_maps(task, source) -> list[np.ndarray]  (early conv layer maps)
    get_logits(task, source)       -> np.ndarray         (pre-softmax scores)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from PIL import Image
import numpy as np

from .model_manager import get_model, get_config, model_available
from .preprocessing import preprocess, preprocess_display


# ──────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────

@dataclass
class PredictionResult:
    task: str
    predicted_class: str
    class_index: int

    # Binary
    probability: Optional[float] = None          # sigmoid output [0,1]

    # Multiclass
    probabilities: Dict[str, float] = field(default_factory=dict)  # label→prob
    logits: Optional[np.ndarray] = None          # raw pre-softmax scores

    # Shared
    is_demo: bool = False                        # True when model file missing


# ──────────────────────────────────────────────
# Demo (mock) results when model files absent
# ──────────────────────────────────────────────

def _demo_binary() -> PredictionResult:
    return PredictionResult(
        task="binary",
        predicted_class="Non-Melanoma",
        class_index=0,
        probability=0.18,
        is_demo=True,
    )


def _demo_multiclass(labels: Dict[int, str]) -> PredictionResult:
    mock = [0.05, 0.08, 0.12, 0.04, 0.15, 0.50, 0.06]
    probs = {labels[i]: mock[i] for i in range(len(labels))}
    return PredictionResult(
        task="multiclass",
        predicted_class=labels[5],
        class_index=5,
        probabilities=probs,
        is_demo=True,
    )


# ──────────────────────────────────────────────
# Main predict function
# ──────────────────────────────────────────────

def predict(
    task: str,
    source: Union[str, np.ndarray, Image.Image],
) -> PredictionResult:
    """
    Run inference for *task* on *source*.

    Args:
        task   : "binary" | "multiclass"
        source : file path, PIL Image, or numpy uint8 array (H, W, 3)

    Returns:
        PredictionResult with probability / probabilities / logits populated.
    """
    cfg = get_config(task)

    if not model_available(task):
        return _demo_binary() if task == "binary" else _demo_multiclass(cfg.labels)

    model = get_model(task)
    img_batch = preprocess(source, cfg.backbone)            # (1, 224, 224, 3)
    raw_output = model.predict(img_batch, verbose=0)        # (1, N)

    if task == "binary":
        probability = float(raw_output[0, 0])
        class_index = 1 if probability > 0.5 else 0
        return PredictionResult(
            task="binary",
            predicted_class=cfg.labels[class_index],
            class_index=class_index,
            probability=probability,
        )
    else:
        import tensorflow as tf
        logits = raw_output[0]                              # (N,)
        probs = tf.nn.softmax(logits).numpy()               # (N,)
        class_index = int(np.argmax(probs))
        prob_dict = {cfg.labels[i]: float(probs[i]) for i in range(len(cfg.labels))}
        return PredictionResult(
            task="multiclass",
            predicted_class=cfg.labels[class_index],
            class_index=class_index,
            probabilities=prob_dict,
            logits=logits,
        )


# ──────────────────────────────────────────────
# Educational helpers
def get_feature_maps(task: str, source, layer_index: int = 10):

    print("\n===== FEATURE MAP DEBUG START =====")

    if not model_available(task):
        print("❌ Model not available")
        return []

    import tensorflow as tf

    model = get_model(task)
    cfg = get_config(task)

    print("Task:", task)
    print("Total top-level layers:", len(model.layers))

    # --------------------------------------------------
    # BINARY → EfficientNet backbone inside wrapper
    # --------------------------------------------------

    if task == "binary":

        print("Binary model detected → EfficientNet")

        backbone = model.get_layer("efficientnetb0")

        conv_layers = [
            l for l in backbone.layers
            if isinstance(l, tf.keras.layers.Conv2D)
            and "expand" in l.name
        ]

        if not conv_layers:
            print("❌ No conv layers in backbone")
            return []

        if layer_index >= len(conv_layers):
            print("❌ layer_index too high")
            return []

        target_layer = backbone.get_layer("block4a_expand_conv")

        print("Using layer:", target_layer.name)

        feat_model = tf.keras.models.Model(
            inputs=backbone.input,
            outputs=target_layer.output,
        )

        img_batch = preprocess(source, cfg.backbone)
        feature_volume = feat_model(img_batch, training=False)[0]

    # --------------------------------------------------
    # MULTICLASS → Flat ResNet model
    # --------------------------------------------------

    elif task == "multiclass":

        print("Multiclass model detected → Flat ResNet")

        conv_layers = [
            l for l in model.layers
            if isinstance(l, tf.keras.layers.Conv2D)
            and "conv" in l.name
        ]

        print("Total ResNet conv layers:", len(conv_layers))

        if not conv_layers:
            print("❌ No conv layers in model")
            return []

        if layer_index >= len(conv_layers):
            print("❌ layer_index too high")
            return []

        target_layer = conv_layers[layer_index]

        print("Using layer:", target_layer.name)

        feat_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=target_layer.output,
        )

        img_batch = preprocess(source, cfg.backbone)
        feature_volume = feat_model(img_batch, training=False)[0]

    else:
        print("❌ Unknown task")
        return []

    print("Feature volume shape:", feature_volume.shape)

    # --------------------------------------------------
    # Normalize maps safely
    # --------------------------------------------------

    maps = []
    num_channels = min(16, feature_volume.shape[-1])

    for c in range(num_channels):
        fm = feature_volume[:, :, c]

        lo = tf.reduce_min(fm)
        hi = tf.reduce_max(fm)
        range_val = hi - lo

        if float(range_val) < 1e-6:
            fm = tf.zeros_like(fm)
        else:
            fm = (fm - lo) / range_val

        maps.append(fm.numpy().astype("float32"))

    print("Generated maps:", len(maps))
    print("===== FEATURE MAP DEBUG END =====\n")

    return maps






def get_logits_and_softmax(
    task: str,
    source: Union[str, np.ndarray, Image.Image],
) -> Optional[Dict[str, np.ndarray]]:
    """
    Return {"logits": array, "softmax": array} for educational display.
    Only meaningful for multiclass; returns None for binary.
    """
    if task != "multiclass" or not model_available(task):
        return None

    import tensorflow as tf

    model = get_model(task)
    cfg = get_config(task)
    img_batch = preprocess(source, cfg.backbone)
    logits = model.predict(img_batch, verbose=0)[0]
    softmax = tf.nn.softmax(logits).numpy()

    return {
        "logits": logits.astype(np.float32),
        "softmax": softmax.astype(np.float32),
        "labels": [cfg.labels[i] for i in range(len(cfg.labels))],
    }