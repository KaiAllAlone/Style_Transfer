# modern_style_transfer_lbgfs.py
# Tested with TensorFlow 2.x (Colab has TF 2.* by default)

import numpy as np
from PIL import Image
from io import BytesIO
import os
from tqdm import trange

import tensorflow as tf
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model
from scipy.optimize import fmin_l_bfgs_b

# -----------------------------
# Hyperparameters
# -----------------------------
ITERATIONS = 10
IMAGE_SIZE = 500
CONTENT_WEIGHT = 0.02
STYLE_WEIGHT = 4.5
TOTAL_VARIATION_WEIGHT = 0.995
TOTAL_VARIATION_LOSS_FACTOR = 1.25  # kept from your code (used in custom TV)
CHANNELS = 3

# Paths (update if needed)
IMAGE_PATH = r"Style_Transfer\Aarohan2k25\buildings.png"
STYLE_PATH = r"Style_Transfer\Aarohan2k25\starry_night.jpg"
INPUT_IMAGE_PATH = "input.png"
STYLE_IMAGE_PATH = "style.png"
OUTPUT_IMAGE_PATH = "output.png"
COMBINED_IMAGE_PATH = "combined.png"

# -----------------------------
# Utilities
# -----------------------------
def load_and_resize(path, size):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return img

def pil_to_vgg_input(pil_img):
    """Return a float32 numpy array shaped (1,H,W,3) preprocessed for VGG16 (i.e. BGR & imagenet mean sub)."""
    arr = np.asarray(pil_img, dtype="float32")
    arr = np.expand_dims(arr, axis=0)
    # Use keras' preprocess_input (it does RGB->BGR and mean subtraction)
    arr = vgg16.preprocess_input(arr.copy())
    return arr

def vgg_input_to_pil(x):
    """Convert a preprocessed VGG input (1,H,W,3) back to PIL RGB."""
    x = x.copy()
    # reverse preprocess_input: the preprocess_input used by vgg16 subtracts imagenet mean and converts RGB->BGR
    # keras vgg16.preprocess_input: x[..., ::-1] then subtract mean
    x = x.reshape((IMAGE_SIZE, IMAGE_SIZE, 3))
    # add back means (these are the VGG16 mean values)
    mean = np.array([103.939, 116.779, 123.68])
    x[..., 0] += mean[0]
    x[..., 1] += mean[1]
    x[..., 2] += mean[2]
    # convert BGR -> RGB
    x = x[..., ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return Image.fromarray(x)

# -----------------------------
# Prepare images
# -----------------------------
input_image = load_and_resize(IMAGE_PATH, IMAGE_SIZE)
style_image = load_and_resize(STYLE_PATH, IMAGE_SIZE)

# save visual copies (optional)
input_image.save(INPUT_IMAGE_PATH)
style_image.save(STYLE_IMAGE_PATH)

# Preprocessed numpy arrays for VGG
content_array = pil_to_vgg_input(input_image)   # shape (1,H,W,3)
style_array   = pil_to_vgg_input(style_image)

# Initialize combination image as random noise (same convention as your original)
x = np.random.uniform(0, 255, (1, IMAGE_SIZE, IMAGE_SIZE, 3)).astype("float32") - 128.0
# Note: We'll pass this x to the optimizer (SciPy expects a flattened array)

# -----------------------------
# Build VGG model to get required layer outputs
# -----------------------------
# Load VGG16 (include_top=False) - weights on imagenet
base_model = vgg16.VGG16(weights="imagenet", include_top=False)
base_model.trainable = False

# Layers used in your original script
style_layer_names = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]
content_layer_name = "block2_conv2"

# Create a model that returns style layer outputs + content layer output
outputs = [base_model.get_layer(name).output for name in style_layer_names + [content_layer_name]]
model = Model(inputs=base_model.input, outputs=outputs)

# -----------------------------
# Helpers to compute gram matrix and losses (TF-friendly)
# -----------------------------
def gram_matrix_tf(x):
    # x: (1, H, W, C)
    x = tf.transpose(x, (0, 3, 1, 2))  # (1, C, H, W)
    b, c, h, w = x.shape
    features = tf.reshape(x, (b, c, h * w))
    gram = tf.matmul(features, features, transpose_b=True)
    return gram

def style_loss_tf(style_feature, comb_feature):
    # Both shapes: (1, H, W, C)
    S = gram_matrix_tf(style_feature)
    C = gram_matrix_tf(comb_feature)
    size = IMAGE_HEIGHT * IMAGE_WIDTH if False else IMAGE_SIZE * IMAGE_SIZE  # keep shape references simple
    # using same formula as your original: sum(square(S-C)) / (4 * (channels^2) * (size^2))
    channels = int(style_feature.shape[-1])
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def content_loss_tf(content_feature, comb_feature):
    return tf.reduce_sum(tf.square(comb_feature - content_feature))

def total_variation_loss_tf(x):
    # x: (1,H,W,3)
    a = tf.square(x[:, :IMAGE_SIZE-1, :IMAGE_SIZE-1, :] - x[:, 1:, :IMAGE_SIZE-1, :])
    b = tf.square(x[:, :IMAGE_SIZE-1, :IMAGE_SIZE-1, :] - x[:, :IMAGE_SIZE-1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, TOTAL_VARIATION_LOSS_FACTOR))

# -----------------------------
# Precompute target features for content and style (as tensors)
# Note: model expects preprocessed input (VGG format)
# -----------------------------
content_tensor = tf.constant(content_array)
style_tensor = tf.constant(style_array)

# run the model once to get features (order matches style_layer_names + [content_layer_name])
style_and_content_features = model(tf.concat([style_tensor, content_tensor], axis=0))
# The previous line returns a list of tensors; but because we concatenated, we need to split per input
# To simplify, we compute separately:
style_feats = model(style_tensor)         # list of style outputs + content output for style image
content_feats = model(content_tensor)     # list of style outputs + content output for content image

# For style, we take first len(style_layer_names) items; content is last item from content_feats
style_targets = [tf.identity(f) for f in style_feats[:len(style_layer_names)]]
content_target = content_feats[-1]

# -----------------------------
# Loss + grad function for SciPy
# -----------------------------
# We will build a TF function that given an input image (preprocessed VGG array) computes loss and grads
@tf.function
def compute_loss_and_grads(x_var):
    """
    x_var: tf.Tensor shape (1,H,W,3), dtype=float32, in VGG-preprocessed space (i.e. using vgg16.preprocess_input convention)
    Returns (loss, grads) as tensors
    """
    with tf.GradientTape() as tape:
        tape.watch(x_var)
        outputs = model(x_var)  # list of layer outputs: style layers then content layer
        # style outputs
        style_outputs = outputs[:len(style_layer_names)]
        content_output = outputs[-1]

        # content loss
        c_loss = CONTENT_WEIGHT * content_loss_tf(content_target, content_output)

        # style loss
        s_loss = 0.0
        for so, st in zip(style_outputs, style_targets):
            s_loss += style_loss_tf(st, so)
        s_loss = (STYLE_WEIGHT / float(len(style_layer_names))) * s_loss

        # total variation loss
        tv = TOTAL_VARIATION_WEIGHT * total_variation_loss_tf(x_var)

        total_loss = c_loss + s_loss + tv

    grads = tape.gradient(total_loss, x_var)
    return total_loss, grads

# SciPy expects functions that accept/return numpy arrays (float64)
def eval_loss_and_grads(x_flat):
    x = x_flat.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3)).astype("float32")
    # Note: the VGG preprocessing (vgg16.preprocess_input) expects inputs already in that form.
    # Our x is already in that space because we initialized x similarly in the original style.
    x_tf = tf.constant(x)
    loss_value, grad_tf = compute_loss_and_grads(x_tf)
    loss_value = loss_value.numpy().astype("float64")
    grad = grad_tf.numpy().astype("float64").flatten()
    return loss_value, grad

# Wrap to match Evaluator API used in many NST examples
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.gradients = None

    def loss(self, x_flat):
        loss_value, grad = eval_loss_and_grads(x_flat)
        self.loss_value = loss_value
        self.gradients = grad
        return self.loss_value

    def grads(self, x_flat):
        return self.gradients

evaluator = Evaluator()

# -----------------------------
# Run L-BFGS
# -----------------------------
x_init = x.flatten().astype("float64")  # initial guess (random noise)
for i in range(ITERATIONS):
    print("Starting iteration", i+1)
    x_opt, min_val, info = fmin_l_bfgs_b(evaluator.loss, x_init, fprime=evaluator.grads, maxfun=20)
    print("Iteration %d completed with loss %d" % (i+1, min_val))
    x_init = x_opt  # continue from the last

# -----------------------------
# Postprocess and save final image
# -----------------------------
x_final = x_opt.reshape((IMAGE_SIZE, IMAGE_SIZE, 3)).astype("float32")
# The x_final is in VGG-preprocessed space (BGR with imagenet mean subtracted) if you used the same initialization.
# Convert back to displayable PIL image:
# Inverse of vgg16.preprocess_input:
#  - add back mean values (in BGR order)
#  - convert BGR -> RGB
mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
# x_final was originally (1,H,W,3) flattened; ensure shape matches
x_disp = x_final.copy()
# If x_init range is similar to original random noise (-128..127), add back mean then clip
x_disp[..., 0] += mean[0]
x_disp[..., 1] += mean[1]
x_disp[..., 2] += mean[2]
# Convert BGR->RGB
x_disp = x_disp[..., ::-1]
x_disp = np.clip(x_disp, 0, 255).astype("uint8")
out_image = Image.fromarray(x_disp)
out_image.save(OUTPUT_IMAGE_PATH)
out_image

# Combined visualization
combined = Image.new("RGB", (IMAGE_SIZE * 3, IMAGE_SIZE))
combined.paste(input_image, (0, 0))
combined.paste(style_image, (IMAGE_SIZE, 0))
combined.paste(out_image, (2 * IMAGE_SIZE, 0))
combined.save(COMBINED_IMAGE_PATH)
print(f"✅ Combined image saved to {COMBINED_IMAGE_PATH}")
combined.show()
