import numpy as np
from tensorflow.keras.preprocessing import image

# Same image size used during training
IMAGE_SIZE = (224, 224)

def preprocess_image(img_file):
    """
    Preprocess image for model inference.

    Args:
        img_file: Uploaded image file (Streamlit file uploader object)

    Returns:
        Preprocessed image tensor ready for prediction
    """
    # Load image
    img = image.load_img(img_file, target_size=IMAGE_SIZE)

    # Convert to array
    img_array = image.img_to_array(img)

    # Normalize (same as training: rescale=1./255)
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
