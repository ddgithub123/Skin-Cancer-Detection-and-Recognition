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
from model.app_controller import get_task



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




import tensorflow as tf
import numpy as np


def compute_gradcam(model, img_batch, class_index=None, task=None):

    debug_backbone(model)

    task = get_task().lower()
    print("\n===== GRAD-CAM START =====")
    print("Task:", task)

    img_batch = tf.convert_to_tensor(img_batch, dtype=tf.float32)

    # ==================================================
    # MULTICLASS (ResNet – flat architecture)
    # ==================================================
    if "multi" in task:

        print("Using ResNet flat model")

        target_layer = model.get_layer("conv5_block3_out")
        print("GradCAM layer:", target_layer.name)

        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[target_layer.output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_batch, training=False)

            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]

            if class_index is None:
                class_index = tf.argmax(predictions[0])

            target = predictions[:, class_index]

        grads = tape.gradient(target, conv_outputs)

    # ==================================================
    # BINARY (EfficientNet – nested backbone)
    # ==================================================
    elif "binary" in task:

        print("→ Binary mode – avoiding sub-model creation to prevent KeyError")

        try:
            backbone = model.get_layer("efficientnetb0")
        except Exception as e:
            print("Cannot find 'efficientnetb0' layer →", e)
            raise

        # Choose best layer (top_activation usually gives nicest maps)
        layer_name = "top_activation"
        try:
            target_layer = backbone.get_layer(layer_name)
            print(f"  Using layer: {layer_name}")
        except:
            layer_name = "top_conv"
            target_layer = backbone.get_layer(layer_name)
            print(f"  Fallback to: {layer_name}")

        # ─── Get activations by calling backbone directly ────────
        # This is safe because backbone is already built/called
        conv_outputs = backbone(img_batch, training=False)
        print("  Backbone activations shape:", conv_outputs.shape)

        # ─── GradientTape – watch the activations tensor ────────
        with tf.GradientTape() as tape:
            tape.watch(conv_outputs)

            # Manual forward through classifier head (same as your test script)
            x = model.get_layer("global_average_pooling2d")(conv_outputs)
            x = model.get_layer("batch_normalization")(x)
            x = model.get_layer("dropout")(x)
            x = model.get_layer("dense")(x)
            x = model.get_layer("batch_normalization_1")(x)
            x = model.get_layer("dropout_1")(x)
            predictions = model.get_layer("dense_1")(x)

            pred_value = predictions[:, 0]
            print(f"  Sigmoid output: {pred_value.numpy().item():.6f}")

            # Same target as test script
            target = tf.math.log(pred_value + 1e-8)

        grads = tape.gradient(target, conv_outputs)

        if grads is None:
            print("!!! GRADIENTS ARE NONE !!! Possible causes:")
            print("  - Model is extremely confident → gradients vanish")
            print("  - Graph disconnection between backbone and head")
            return np.zeros((7, 7), dtype=np.float32)  # small fallback

        print("  Gradients computed | shape:", grads.shape)
        print(f"  Grad mean/std/max: {tf.reduce_mean(grads):.3e}  {tf.math.reduce_std(grads):.3e}  {tf.reduce_max(tf.abs(grads)):.3e}")

    # ==================================================
    # BUILD HEATMAP (Unified for both)
    # ==================================================
    conv_outputs = conv_outputs[0]
    grads = grads[0]

    # Positive gradients only for cleaner map
    grads = tf.nn.relu(grads)

    weights = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)
    heatmap = tf.nn.relu(heatmap)

    heatmap = heatmap.numpy()
    max_val = heatmap.max() if heatmap.size > 0 else 0

    # ──── Add / uncomment these lines ────────────────────────────────

    print("━"*60)
    print("DIAGNOSTIC ─ BINARY GRAD-CAM")
    print("Task                  :", task)
    print("Prediction (sigmoid)  :", float(predictions[0,0]) if 'predictions' in locals() else "—")
    print("Target value (log)    :", target.numpy().item() if 'target' in locals() else "—")
    print("Conv outputs shape    :", conv_outputs.shape)
    print("Gradients shape       :", grads.shape if grads is not None else "None!")
    print("Grad mean / std / max :", 
        float(tf.reduce_mean(grads)) if grads is not None else "—",
        float(tf.math.reduce_std(grads)) if grads is not None else "—",
        float(tf.reduce_max(tf.abs(grads))) if grads is not None else "—")
    print("Heatmap before norm   : min =", heatmap.min(), "max =", heatmap.max())
    print("━"*60)

    if max_val > 1e-8:
        heatmap /= max_val
    else:
        heatmap.fill(0)

    print("===== GRAD-CAM SUCCESS =====\n")
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