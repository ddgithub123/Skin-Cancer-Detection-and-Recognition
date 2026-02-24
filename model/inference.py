import os
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("TensorFlow not found. Inference will be disabled.")

from .preprocess import preprocess_image
import cv2

# Load models once if TF is available
binary_model = None
multiclass_model = None
last_conv_layer_name = None

# Labels for HAM10000 Multiclass
MULTICLASS_LABELS = {
    0: "Actinic keratoses (akiec)",
    1: "Basal cell carcinoma (bcc)",
    2: "Benign keratosis-like lesions (bkl)",
    3: "Dermatofibroma (df)",
    4: "Melanoma (mel)",
    5: "Melanocytic nevi (nv)",
    6: "Vascular lesions (vasc)"
}

if HAS_TF:
    try:
        # Load Binary Model
        BINARY_MODEL_PATH = os.path.join(os.path.dirname(__file__), "binary_model.h5")
        if os.path.exists(BINARY_MODEL_PATH):
            binary_model = load_model(BINARY_MODEL_PATH)
        
        # Load Multiclass Model (if exists)
        MULTI_MODEL_PATH = os.path.join(os.path.dirname(__file__), "multiclass_model.h5")
        if os.path.exists(MULTI_MODEL_PATH):
            multiclass_model = load_model(MULTI_MODEL_PATH)
        
        # Get last convolution layer name from binary model (for Grad-CAM)
        def get_last_conv_layer(m):
            if not m: return None
            for layer in reversed(m.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    return layer.name
            return None

        # Use binary model as primary for Grad-CAM architecture reference
        last_conv_layer_name = get_last_conv_layer(binary_model)
    except Exception as e:
        print(f"Error loading models: {e}")
        HAS_TF = False


def predict_binary(img_file):
    if not HAS_TF or binary_model is None:
        return {
            "class": "Demo Result (Non-Melanoma)",
            "probability": 0.15,
            "error": "TensorFlow not available or model missing."
        }
    
    processed_img = preprocess_image(img_file)
    prediction = binary_model.predict(processed_img)

    probability = float(prediction[0][0])
    predicted_class = 1 if probability > 0.5 else 0
    class_name = "Melanoma" if predicted_class == 1 else "Non-Melanoma"

    return {
        "class": class_name,
        "probability": probability
    }


def predict_multiclass(img_file):
    if not HAS_TF or multiclass_model is None:
        # Mock results for demo
        mock_probs = [0.05, 0.05, 0.1, 0.05, 0.15, 0.55, 0.05]
        return {
            "class": MULTICLASS_LABELS[5],
            "probabilities": {MULTICLASS_LABELS[i]: mock_probs[i] for i in range(7)}
        }
    
    processed_img = preprocess_image(img_file)
    prediction = multiclass_model.predict(processed_img)[0]
    
    predicted_idx = np.argmax(prediction)
    probabilities = {MULTICLASS_LABELS[i]: float(prediction[i]) for i in range(7)}
    
    return {
        "class": MULTICLASS_LABELS[predicted_idx],
        "probabilities": probabilities
    }

def generate_gradcam(img_path, model_type="binary"):

    if not HAS_TF:
        return np.zeros((224, 224))

    active_model = binary_model if model_type == "binary" else multiclass_model
    if active_model is None:
        return np.zeros((224, 224))

    # Load & preprocess image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = img.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 🔥 IMPORTANT: Use fixed ResNet50 conv layer
    last_conv_layer_name = "conv5_block3_out"

    grad_model = tf.keras.models.Model(
        [active_model.inputs],
        [active_model.get_layer(last_conv_layer_name).output, active_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array])

        if isinstance(predictions, list):
            predictions = predictions[0]

        if model_type == "binary":
            loss = predictions[:, 0]
        else:
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    # 🔥 SAFE weighted sum instead of matrix multiply
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)

    if max_val > 0:
        heatmap /= max_val

    heatmap = heatmap.numpy()

    return heatmap

def overlay_gradcam(original_image_path, heatmap, alpha=0.4):

    img = cv2.imread(original_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return overlay