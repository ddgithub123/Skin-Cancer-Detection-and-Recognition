import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
MODEL_PATH = ".\\model\\binary_model_converted.keras"
IMG_PATH   = ".\\test_images\\3.png"
IMG_SIZE   = 224


# ===============================
# LOAD MODEL
# ===============================
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Warm-up forward pass
dummy = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
_ = model(dummy, training=False)

print("Model loaded successfully ✔")


# ===============================
# PREPROCESS IMAGE (Correct for EfficientNet)
# ===============================
img = image.load_img(IMG_PATH, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array.astype("float32"))

print("Image shape:", img_array.shape)


# ===============================
# BACKBONE EXTRACTION
# ===============================
backbone = model.get_layer("efficientnetb0")

# Try slightly earlier conv layer for better localization
target_layer_name = "top_conv"
target_layer = backbone.get_layer(target_layer_name)

backbone_model = tf.keras.Model(
    inputs=backbone.input,
    outputs=target_layer.output
)

print(f"Using Grad-CAM layer: {target_layer_name}")


# ===============================
# GRAD-CAM
# ===============================
with tf.GradientTape() as tape:

    conv_outputs = backbone_model(img_array, training=False)
    tape.watch(conv_outputs)

    # Forward through classifier head manually
    x = model.get_layer("global_average_pooling2d")(conv_outputs)
    x = model.get_layer("batch_normalization")(x)
    x = model.get_layer("dropout")(x)
    x = model.get_layer("dense")(x)
    x = model.get_layer("batch_normalization_1")(x)
    x = model.get_layer("dropout_1")(x)
    predictions = model.get_layer("dense_1")(x)

    # Binary target stabilization
    pred_value = predictions[:, 0]
    print("Prediction score:", float(pred_value[0]))

    target = tf.math.log(pred_value + 1e-8)

grads = tape.gradient(target, conv_outputs)

if grads is None:
    raise RuntimeError("Gradients are None ❌")

print("Gradients OK ✔")


# ===============================
# BUILD HEATMAP
# ===============================
# Only positive gradients (sharper maps)
conv_outputs = conv_outputs[0]
grads = grads[0]

weights = tf.reduce_mean(grads, axis=(0, 1))
heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)
heatmap = tf.nn.relu(heatmap)

heatmap = heatmap.numpy()

if heatmap.max() > 1e-8:
    heatmap /= heatmap.max()
else:
    heatmap.fill(0)

heatmap = heatmap.astype(np.float32)

print("Heatmap shape:", heatmap.shape)


# ===============================
# OVERLAY
# ===============================
original = cv2.imread(IMG_PATH)
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))

heatmap_resized = cv2.resize(
    heatmap,
    (IMG_SIZE, IMG_SIZE),
    interpolation=cv2.INTER_LINEAR
)

heatmap_u8 = np.uint8(255 * heatmap_resized)

heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

plt.figure(figsize=(6, 6))
plt.imshow(overlay)
plt.title("Binary Grad-CAM")
plt.axis("off")
plt.show()

print("Grad-CAM complete ✔")