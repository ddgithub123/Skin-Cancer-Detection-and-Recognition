import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
MODEL_PATH = r".\\model\\binary_model.h5"   # change if needed
IMAGE_PATH = r".\\test_images\\1.png"
IMG_SIZE = (224, 224)

# ===============================
# 1. LOAD MODEL
# ===============================
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")

# ===============================
# 2. LOAD AND PREPROCESS IMAGE
# ===============================
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, IMG_SIZE)

img_array = img.astype("float32") / 255.0
img_array = np.expand_dims(img_array, axis=0)

# ===============================
# 3. FIND LAST CONV LAYER
# ===============================
last_conv_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer.name
        break

if last_conv_layer is None:
    raise ValueError("No Conv2D layer found in model.")

print("Last Conv Layer:", last_conv_layer)

# ===============================
# 4. BUILD GRAD MODEL
# ===============================
grad_model = tf.keras.models.Model(
    [model.inputs],
    [model.get_layer(last_conv_layer).output, model.output]
)

# ===============================
# 5. COMPUTE GRAD-CAM
# ===============================
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)

    # If model output is wrapped in list, unwrap it
    if isinstance(predictions, list):
        predictions = predictions[0]

    class_index = tf.argmax(predictions[0])
    loss = predictions[:, class_index]

grads = tape.gradient(loss, conv_outputs)

# Global average pooling of gradients
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# Normalize heatmap
heatmap = tf.maximum(heatmap, 0)
heatmap /= tf.reduce_max(heatmap)
heatmap = heatmap.numpy()

# ===============================
# 6. OVERLAY HEATMAP
# ===============================
heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap_resized = np.uint8(255 * heatmap_resized)

heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

# ===============================
# 7. DISPLAY RESULTS
# ===============================
plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Grad-CAM Heatmap")
plt.imshow(heatmap, cmap='jet')
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()