import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image, ImageOps
from tqdm import trange
import numpy as np
import os
import cv2 as cv
# -----------------------------
# Settings (tweakable)
# -----------------------------
IMAGE_SIZE = 500
ITERATIONS = 200
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT = 1e5      # tuned for current gram normalization
TOTAL_VARIATION_WEIGHT = 1e-6
START_FROM_CONTENT = True

CONTENT_IMAGE_PATH = r"Style_Transfer\Aarohan2k25\buildings.png"
STYLE_IMAGE_PATH   = r"Style_Transfer\Aarohan2k25\starry_night.jpg"
OUTPUT_PATH = "output.png"
COMBINED_PATH = "combined.png"
INTERMEDIATE_DIR = "intermediates"
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Helpers
# -----------------------------
def load_image(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)  # [0,1]
    return tensor

def save_image(tensor, path):
    img = tensor.detach().cpu().clamp(0,1).squeeze(0)
    pil = transforms.ToPILImage()(img)
    pil.save(path)

def make_combined(content_t, style_t, out_t, path=COMBINED_PATH):
    c = transforms.ToPILImage()(content_t.detach().cpu().squeeze(0).clamp(0,1))
    s = transforms.ToPILImage()(style_t.detach().cpu().squeeze(0).clamp(0,1))
    o = transforms.ToPILImage()(out_t.detach().cpu().squeeze(0).clamp(0,1))
    combined = Image.new("RGB", (IMAGE_SIZE * 3, IMAGE_SIZE))
    combined.paste(c, (0,0)); combined.paste(s, (IMAGE_SIZE,0)); combined.paste(o, (IMAGE_SIZE*2,0))
    combined.save(path)

# -----------------------------
# Normalization for torchvision VGG
# -----------------------------
cnn_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_std  = torch.tensor([0.229, 0.224, 0.225]).to(device)

def normalize_batch(batch):
    return (batch - cnn_mean[None,:,None,None]) / cnn_std[None,:,None,None]

# -----------------------------
# Gram matrix
# -----------------------------
def gram_matrix(input):
    # input shape: (1, C, H, W)
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)   # (C, H*W) when b==1
    G = torch.mm(features, features.t())  # (C, C)
    # normalize by (C * H * W) - keep consistent with prior code
    return G / (c * h * w)

# -----------------------------
# Total variation
# -----------------------------
def total_variation_loss(img):
    # img shape: (1, C, H, W)
    x_diff = img[:, :, :, 1:] - img[:, :, :, :-1]
    y_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
    return torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))

# -----------------------------
# Load images
# -----------------------------
content = load_image(CONTENT_IMAGE_PATH, IMAGE_SIZE)
style   = load_image(STYLE_IMAGE_PATH, IMAGE_SIZE)

if START_FROM_CONTENT:
    combination = content.clone().detach().requires_grad_(True)
else:
    combination = torch.randn_like(content, device=device, requires_grad=True)

# -----------------------------
# Load VGG19 (features only)
# -----------------------------
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
for p in vgg.parameters():
    p.requires_grad = False

# layer indices mapping in torchvision VGG19 features:
# conv1_1=0, conv2_1=5, conv3_1=10, conv4_1=19, conv4_2=21 (content), conv5_1=28
STYLE_LAYERS = ['0','5','10','19','28']
CONTENT_LAYER = '21'

def extract_features(x, model, layers):
    """Return dict of activations for requested layer indices (strings)."""
    features = {}
    out = normalize_batch(x)
    for name, layer in model._modules.items():
        out = layer(out)
        if name in layers:
            features[name] = out
    return features

# -----------------------------
# Optimizer: ADAM with small lr
# -----------------------------
# KEY: Adam needs a small lr here for stable updates
optimizer = optim.Adam([combination], lr=0.02)   # <<<<< LR tuned down from 10 -> 0.02

# -----------------------------
# Main optimization loop
# -----------------------------
print("Starting style transfer... (this may take a while)")

# Precompute features for content & style (normalized inside extract_features)
content_features = extract_features(content, vgg, [CONTENT_LAYER])
style_features = extract_features(style, vgg, STYLE_LAYERS)

# Precompute style gram matrices
style_grams = {layer: gram_matrix(style_features[layer]) for layer in STYLE_LAYERS}

for i in trange(ITERATIONS, desc="Style Transfer"):
    optimizer.zero_grad()

    combo_feats = extract_features(combination, vgg, STYLE_LAYERS + [CONTENT_LAYER])

    # content loss (MSE)
    c_loss = F.mse_loss(combo_feats[CONTENT_LAYER], content_features[CONTENT_LAYER]) * CONTENT_WEIGHT

    # style loss (sum over selected layers)
    s_loss = 0.0
    for layer in STYLE_LAYERS:
        gm_comb = gram_matrix(combo_feats[layer])
        gm_style = style_grams[layer]
        s_loss += F.mse_loss(gm_comb, gm_style)
    s_loss = (s_loss / len(STYLE_LAYERS)) * STYLE_WEIGHT

    # total variation
    tv_loss = TOTAL_VARIATION_WEIGHT * total_variation_loss(combination)

    loss = c_loss + s_loss + tv_loss
    loss.backward()
    optimizer.step()

    # clip to valid [0,1] range after step (important)
    with torch.no_grad():
        combination.clamp_(0.0, 1.0)

    # save intermediate images occasionally so you can inspect progress
    if (i % 50 == 0) or (i == ITERATIONS-1):
        inter_path = os.path.join(INTERMEDIATE_DIR, f"iter_{i+1:04d}.png")
        save_image(combination, inter_path)
        print(f"[iter {i+1:03d}] loss={loss.item():.2f} -> saved {inter_path}")

# -----------------------------
# Save final images
# -----------------------------
save_image(combination, OUTPUT_PATH)
make_combined(content, style, combination, COMBINED_PATH)
cv.imshow("Output", cv.imread(OUTPUT_PATH))
cv.waitKey(0)
print("Finished. Output saved to:", OUTPUT_PATH, COMBINED_PATH)
