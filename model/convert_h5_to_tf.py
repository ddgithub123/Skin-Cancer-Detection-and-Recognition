import tensorflow as tf

# =========================
# CONFIG
# =========================
H5_PATH = "binary_model.h5"
KERAS_PATH = "binary_model_converted.keras"

print("Loading H5 model...")
model = tf.keras.models.load_model(H5_PATH, compile=False)
print("Model loaded ✔")

# Warm-up forward pass
print("Running warm-up forward pass...")
dummy = tf.zeros((1, 224, 224, 3))
_ = model(dummy, training=False)
print("Warm-up complete ✔")

# Save in new Keras v3 format
print("Saving as .keras format...")
model.save(KERAS_PATH)

print("Model successfully saved as:", KERAS_PATH)
print("Done ✔")