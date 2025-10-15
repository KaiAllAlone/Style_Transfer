import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# =====================================================
# Configuration
# =====================================================
STYLE_PATH = "Programs\Aarohan2k25\god.jpg"
MAX_DIM = 256  # Keep small for real-time speed
USE_MIXED_PRECISION = True  # Set False if GPU doesn't support

# =====================================================
# Optional: Mixed Precision for faster GPU
# =====================================================
if USE_MIXED_PRECISION:
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"✓ Mixed precision enabled: {policy.name}")

# =====================================================
# Functions
# =====================================================
def load_image(img_path, max_dim=None):
    """Load and preprocess image as a float32 tensor."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if max_dim:
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
    img = tf.expand_dims(img, axis=0)
    return img

# =====================================================
# Main Program
# =====================================================
def main():
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU found: {gpus[0].name}")
    else:
        print("⚠️ No GPU found, running on CPU")

    # Load model (cached)
    print("Loading style transfer model...")
    model_url = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    model = hub.load(model_url)
    print("✓ Model loaded")

    # Preprocess and cache style image
    style_image = load_image(STYLE_PATH, max_dim=MAX_DIM)
    print(f"✓ Style image shape: {style_image.shape}")

    # Precompile the model call for faster inference
    @tf.function
    def stylize(content, style):
        return model(content, style)[0]

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Could not open webcam")
        return

    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB float32 tensor
        content_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        content_image = tf.image.convert_image_dtype(content_image, tf.float32)
        content_image = tf.expand_dims(content_image, axis=0)

        # Apply style transfer (tf.function compiled)
        stylized_image = stylize(content_image, style_image)

        # Convert tensor to uint8 for OpenCV
        stylized_image_np = np.squeeze(stylized_image.numpy())
        stylized_image_np = np.clip(stylized_image_np * 255, 0, 255).astype(np.uint8)
        stylized_image_np = cv2.cvtColor(stylized_image_np, cv2.COLOR_RGB2BGR)

        # Display the stylized video
        cv2.imshow('Stylized Webcam', stylized_image_np)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✓ Live style transfer ended")

# =====================================================
if __name__ == "__main__":
    main()
