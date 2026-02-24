"""
gradcam.py
──────────
Stable architecture-agnostic Grad-CAM.

Works with:
• EfficientNet
• ResNet
• Nested sub-models
• Binary + Multiclass
"""

from __future__ import annotations
import numpy as np
import cv2
from typing import Optional
import tensorflow as tf



# ──────────────────────────────────────────────
# Layer detection
# ──────────────────────────────────────────────

def find_last_conv_layer(model):
    import tensorflow as tf

    def recursive_search(m):
        for layer in reversed(m.layers):

            # If this layer is itself a model (like EfficientNet)
            if hasattr(layer, "layers") and len(layer.layers) > 0:
                result = recursive_search(layer)
                if result is not None:
                    return result

            if isinstance(layer, (
                tf.keras.layers.Conv2D,
                tf.keras.layers.DepthwiseConv2D
            )):
                return layer

        return None

    layer = recursive_search(model)

    if layer is None:
        raise ValueError("No Conv2D layer found in model.")

    return layer

# ──────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────

import tensorflow as tf

def debug_backbone(model):

    print("\n====== MODEL DEBUG START ======")

    print("\nTop-level layers:")
    for layer in model.layers:
        print(" -", layer.name, "|", type(layer))

    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break

    if backbone is None:
        print("\n❌ No backbone model found.")
        return

    print("\nBackbone detected:", backbone.name)
    print("Backbone type:", type(backbone))
    print("Backbone layer count:", len(backbone.layers))

    print("\nListing last 20 backbone layers:")
    for layer in backbone.layers[-20:]:
        print(" -", layer.name, "|", type(layer))

    print("\nChecking if 'top_conv' exists:")
    try:
        l = backbone.get_layer("top_conv")
        print("✅ top_conv exists:", l)
    except Exception as e:
        print("❌ top_conv NOT found:", e)

    print("\nChecking if backbone output shape:")
    print(backbone.output_shape)

    print("\n====== MODEL DEBUG END ======\n")






def compute_gradcam(model, img_batch, class_index=None):

    debug_backbone(model)

    img_batch = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    # Identify backbone
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break

    if backbone is None:
        raise ValueError("Backbone not found in model.")

    backbone_name = backbone.name.lower()

    # Hardcode correct conv layer
    if "efficientnet" in backbone_name:
        target_layer = backbone.get_layer("top_conv")
    elif "resnet" in backbone_name:
        target_layer = backbone.get_layer("conv5_block3_out")
    else:
        raise ValueError("Unsupported backbone.")

    print("Using Grad-CAM layer:", target_layer.name)

    # Build grad model using nested layer correctly
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch, training=False)

        if predictions.shape[-1] == 1:
            target = predictions[:, 0]
        else:
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            target = predictions[:, class_index]

    grads = tape.gradient(target, conv_outputs)

    if grads is None:
        raise RuntimeError("Gradients are None.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)

    heatmap = heatmap.numpy()
    max_val = heatmap.max()

    if max_val > 1e-8:
        heatmap /= max_val
    else:
        heatmap[:] = 0

    return heatmap.astype(np.float32)
















# ──────────────────────────────────────────────
# Overlay
# ──────────────────────────────────────────────

def overlay_heatmap(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:

    h, w = original_rgb.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        original_rgb.astype(np.uint8),
        1 - alpha,
        heatmap_color,
        alpha,
        0
    )

    return overlay