# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import trange

# -----------------------------
# Parameters (adjustable)
# -----------------------------
IMAGE_SIZE = 500
ITERATIONS = 15
CONTENT_WEIGHT = 0.02
STYLE_WEIGHT = 15.0
TOTAL_VARIATION_WEIGHT = 1.0
START_FROM_CONTENT = True  # False will start from noise

# -----------------------------
# Hardcoded image paths
# -----------------------------
CONTENT_IMAGE_PATH = r"Style_Transfer\Aarohan2k25\buildings.png"   # Replace with your content image path
STYLE_IMAGE_PATH = r"Style_Transfer\Aarohan2k25\starry_night.jpg"     # Replace with your style image path

# -----------------------------
# Helper functions
# -----------------------------
def load_and_process_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return tf.convert_to_tensor(image, dtype=tf.float32)

def deprocess_image(x):
    x = x.numpy().reshape((IMAGE_SIZE, IMAGE_SIZE, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return Image.fromarray(x)

def gram_matrix(tensor):
    tensor = tf.transpose(tensor, (2, 0, 1))
    features = tf.reshape(tensor, (tensor.shape[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

# -----------------------------
# Load images
# -----------------------------
content_image = load_and_process_image(CONTENT_IMAGE_PATH)
style_image = load_and_process_image(STYLE_IMAGE_PATH)

# Start from content image or random noise
if START_FROM_CONTENT:
    combination_image = tf.Variable(content_image)
else:
    combination_image = tf.Variable(tf.random.uniform(content_image.shape, 0, 255))

# -----------------------------
# Load VGG16 model
# -----------------------------
vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
vgg.trainable = False

STYLE_LAYERS = ["block1_conv2","block2_conv2","block3_conv3","block4_conv3","block5_conv3"]
CONTENT_LAYER = "block2_conv2"

style_outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS]
content_output = vgg.get_layer(CONTENT_LAYER).output
model_outputs = style_outputs + [content_output]
model = tf.keras.Model(vgg.input, model_outputs)

# -----------------------------
# Loss functions
# -----------------------------
def compute_loss(combination):
    outputs = model(combination)
    style_outputs_comb = outputs[:len(STYLE_LAYERS)]
    content_output_comb = outputs[len(STYLE_LAYERS)]

    # Content loss
    content_features = model(content_image)[len(STYLE_LAYERS)]
    c_loss = CONTENT_WEIGHT * tf.reduce_sum(tf.square(content_output_comb - content_features))

    # Style loss
    s_loss = 0
    style_features = model(style_image)[:len(STYLE_LAYERS)]
    for comb, style in zip(style_outputs_comb, style_features):
        s_loss += tf.reduce_sum(tf.square(gram_matrix(comb[0]) - gram_matrix(style[0])))
    s_loss *= STYLE_WEIGHT / len(STYLE_LAYERS)

    # Total variation loss
    tv_loss = TOTAL_VARIATION_WEIGHT * tf.image.total_variation(combination)

    total_loss = c_loss + s_loss + tv_loss
    return total_loss

# -----------------------------
# Optimization
# -----------------------------
optimizer = tf.optimizers.Adam(learning_rate=10.0)

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image)
    grads = tape.gradient(loss, combination_image)
    optimizer.apply_gradients([(grads, combination_image)])
    combination_image.assign(tf.clip_by_value(combination_image, -128.0, 128.0))
    return loss

# -----------------------------
# Run style transfer
# -----------------------------
for i in trange(ITERATIONS, desc="Style Transfer"):
    loss = train_step()
    if i % 5 == 0 or i == ITERATIONS-1:
        print(f"Iteration {i+1}/{ITERATIONS}, Loss: {loss.numpy().item():.2f}")


# -----------------------------
# Save result
# -----------------------------
output_image = deprocess_image(combination_image)
output_image.save("output.png")
output_image.show()

# -----------------------------
# Combined visualization
# -----------------------------
def create_combined_image(content_img, style_img, output_img, path="combined.png"):
    combined_width = IMAGE_SIZE * 3
    combined_image = Image.new("RGB", (combined_width, IMAGE_SIZE))
    x_offset = 0
    for img in [content_img, style_img, output_img]:
        combined_image.paste(img, (x_offset, 0))
        x_offset += IMAGE_SIZE
    combined_image.save(path)
    combined_image.show()

content_display = deprocess_image(content_image)
style_display = deprocess_image(style_image)
create_combined_image(content_display, style_display, output_image)
