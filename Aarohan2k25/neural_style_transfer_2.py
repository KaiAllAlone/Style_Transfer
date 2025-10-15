import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Enable mixed precision for faster GPU computation
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

import tensorflow_hub as hub
from matplotlib import pyplot as plt
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import time

# =====================================================
# Configuration
# =====================================================
CONTENT_PATH = "Programs/Aarohan2k25/WIN_20251015_15_02_39_Pro.jpg"
STYLE_PATH = "Programs/Aarohan2k25/starry_night.jpg"
OUTPUT_PATH = "generated_img.jpg"

# Model configuration
MODEL_CACHE_DIR = "models/style_transfer_model"
MODEL_URL = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'

# Performance settings
MAX_DIM = 512  # Reduce for faster processing (256, 384, 512)
USE_GPU = True  # Set to False to force CPU
ENABLE_XLA = True  # XLA compilation for faster execution

# =====================================================
# GPU Configuration
# =====================================================
def configure_gpu():
    """Configure GPU settings for optimal performance."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to prevent OOM errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable XLA (Accelerated Linear Algebra)
            if ENABLE_XLA:
                tf.config.optimizer.set_jit(True)
            
            print(f"✓ GPU detected: {len(gpus)} device(s)")
            return True
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return False
    else:
        print("⚠ No GPU detected, using CPU")
        return False

# =====================================================
# Optimized Functions
# =====================================================
@tf.function(jit_compile=ENABLE_XLA)
def preprocess_image(img, max_dim):
    """JIT-compiled image preprocessing for speed."""
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    if max_dim:
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = tf.reduce_max(shape)
        scale = tf.cast(max_dim, tf.float32) / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape, method='bicubic')
    
    return tf.expand_dims(img, axis=0)

def load_image_fast(img_path, max_dim=None):
    """
    Optimized image loading with parallel I/O.
    
    Args:
        img_path: Path to the image file
        max_dim: Maximum dimension for resizing
    
    Returns:
        Preprocessed image tensor
    """
    try:
        # Use tf.data for efficient I/O
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = preprocess_image(img, max_dim)
        return img
        
    except Exception as e:
        print(f"Error loading image from {img_path}: {e}")
        raise

def load_model_cached(cache_dir=MODEL_CACHE_DIR):
    """
    Load model from cache or download and cache it.
    
    Args:
        cache_dir: Directory to cache the model
    
    Returns:
        Loaded and optimized model
    """
    try:
        if os.path.exists(cache_dir):
            print(f"✓ Loading model from cache: {cache_dir}")
            model = hub.load(cache_dir)
        else:
            print(f"Downloading model (first time only)...")
            os.makedirs(os.path.dirname(cache_dir) if os.path.dirname(cache_dir) else ".", exist_ok=True)
            
            # Download model
            model = hub.load(MODEL_URL)
            
            # Save to cache
            tf.saved_model.save(model, cache_dir)
            print(f"✓ Model cached at: {cache_dir}")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@tf.function(reduce_retracing=True)
def apply_style_transfer(model, content_image, style_image):
    """
    JIT-compiled style transfer for maximum speed.
    
    Args:
        model: Style transfer model
        content_image: Content image tensor
        style_image: Style image tensor
    
    Returns:
        Stylized image tensor
    """
    return model(content_image, style_image)[0]

def save_image_fast(img_tensor, output_path):
    """
    Optimized image saving with parallel I/O.
    
    Args:
        img_tensor: Image tensor to save
        output_path: Path where image will be saved
    """
    try:
        img = np.squeeze(img_tensor)
        
        if hasattr(img, 'numpy'):
            img = img.numpy()
        
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Use OpenCV's optimized writing
        cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"✓ Stylized image saved: {output_path}")
        
    except Exception as e:
        print(f"Error saving image: {e}")
        raise

def visualize_images_fast(content_img, style_img, stylized_img, save_comparison=True):
    """Fast visualization with optional saving."""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        images = [content_img, style_img, stylized_img]
        titles = ['Content Image', 'Style Image', 'Stylized Result']
        
        for ax, img, title in zip(axes, images, titles):
            display_img = np.squeeze(img)
            if hasattr(display_img, 'numpy'):
                display_img = display_img.numpy()
            ax.imshow(display_img)
            ax.axis('off')
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_comparison:
            comparison_path = OUTPUT_PATH.replace('.jpg', '_comparison.jpg')
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            print(f"✓ Comparison saved: {comparison_path}")
        
        plt.show()
        plt.close(fig)
        
    except Exception as e:
        print(f"Warning: Could not display images: {e}")

def validate_paths(*paths):
    """Validate that all input files exist."""
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

# =====================================================
# Main Program
# =====================================================
def main():
    print("=" * 60)
    print("⚡ Optimized Neural Style Transfer")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Configure GPU
        print("\n[1/6] Configuring hardware...")
        has_gpu = configure_gpu()
        if not USE_GPU and has_gpu:
            tf.config.set_visible_devices([], 'GPU')
            print("✓ GPU disabled (using CPU)")
        
        # Validate input files
        print("\n[2/6] Validating input files...")
        validate_paths(CONTENT_PATH, STYLE_PATH)
        print("✓ Input files found")
        
        # Load model (cached)
        print("\n[3/6] Loading model...")
        model_start = time.time()
        model = load_model_cached()
        print(f"✓ Model loaded in {time.time() - model_start:.2f}s")
        
        # Load images in parallel
        print("\n[4/6] Loading images...")
        img_start = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            content_future = executor.submit(load_image_fast, CONTENT_PATH, MAX_DIM)
            style_future = executor.submit(load_image_fast, STYLE_PATH, MAX_DIM)
            
            content_image = content_future.result()
            style_image = style_future.result()
        
        print(f"✓ Images loaded in {time.time() - img_start:.2f}s")
        print(f"  Content: {content_image.shape}")
        print(f"  Style: {style_image.shape}")
        
        # Perform style transfer (JIT-compiled)
        print("\n[5/6] Applying style transfer...")
        transfer_start = time.time()
        
        # Warm-up run for JIT compilation
        if ENABLE_XLA:
            _ = apply_style_transfer(model, content_image, style_image)
            print("✓ Model compiled (first run)")
        
        # Actual transfer
        stylized_image = apply_style_transfer(model, content_image, style_image)
        transfer_time = time.time() - transfer_start
        print(f"✓ Style transfer completed in {transfer_time:.2f}s")
        
        # Save and visualize
        print("\n[6/6] Saving results...")
        save_start = time.time()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            save_future = executor.submit(save_image_fast, stylized_image, OUTPUT_PATH)
            viz_future = executor.submit(visualize_images_fast, content_image, 
                                        style_image, stylized_image, True)
            
            save_future.result()
            viz_future.result()
        
        print(f"✓ Results saved in {time.time() - save_start:.2f}s")
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"⚡ Total time: {total_time:.2f}s")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Please check your file paths and try again.")
        
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
        import traceback
        traceback.print_exc()

# =====================================================
# Batch Processing (Bonus Feature)
# =====================================================
def batch_process(content_paths, style_path, output_dir="output"):
    """
    Process multiple images with the same style.
    
    Args:
        content_paths: List of content image paths
        style_path: Path to style image
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🔥 Batch processing {len(content_paths)} images...")
    
    # Load model once
    model = load_model_cached()
    style_image = load_image_fast(style_path, MAX_DIM)
    
    for i, content_path in enumerate(content_paths):
        print(f"\nProcessing image {i+1}/{len(content_paths)}...")
        
        content_image = load_image_fast(content_path, MAX_DIM)
        stylized_image = apply_style_transfer(model, content_image, style_image)
        
        output_path = os.path.join(output_dir, f"stylized_{i+1}.jpg")
        save_image_fast(stylized_image, output_path)
    
    print(f"\n✓ Batch processing complete! Results in: {output_dir}")

# =====================================================
# Entry Point
# =====================================================
if __name__ == "__main__":
    main()
    
    # Example batch processing (uncomment to use):
    # batch_process(
    #     content_paths=["image1.jpg", "image2.jpg", "image3.jpg"],
    #     style_path=STYLE_PATH,
    #     output_dir="batch_output"
    # )