import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from .preprocess import preprocess_image
import cv2

# Load model once
MODEL_PATH = os.path.join(os.path.dirname(__file__), "binary_model.h5")
model = load_model(MODEL_PATH)

# Get last convolution layer automatically
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")

last_conv_layer_name = get_last_conv_layer(model)


def predict_binary(img_file):
    processed_img = preprocess_image(img_file)
    prediction = model.predict(processed_img)

    probability = float(prediction[0][0])
    predicted_class = 1 if probability > 0.5 else 0
    class_name = "Melanoma" if predicted_class == 1 else "Non-Melanoma"

    return {
        "class": class_name,
        "probability": probability
    }


def generate_gradcam(img_file):
    img = preprocess_image(img_file)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap


def overlay_gradcam(original_image_path, heatmap, alpha=0.4):
    img = cv2.imread(original_image_path)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    return superimposed_img
