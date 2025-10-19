import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image, ImageOps
from tqdm import trange
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt

#hyperparameters
IMAGE_SIZE = 500
ITERATIONS = 200
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT = 1e8      
TOTAL_VARIATION_WEIGHT = 1e-8
START_FROM_CONTENT = True

CONTENT_IMAGE_PATH = r"Style_Transfer\Aarohan2k25\cat.jpg"
STYLE_IMAGE_PATH   = r"Style_Transfer\Aarohan2k25\starry_night.jpg"
OUTPUT_PATH = "output.png"
COMBINED_PATH = "combined.png"
INTERMEDIATE_DIR = "intermediates"
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


cnn_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_std  = torch.tensor([0.229, 0.224, 0.225]).to(device)

def normalize_batch(batch):
    return (batch - cnn_mean[None,:,None,None]) / cnn_std[None,:,None,None]


def gram_matrix(input):
    # input shape: (1, C, H, W)
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)   # (C, H*W) when b==1
    G = torch.mm(features, features.t())  # (C, C)
    return G / (c * h * w)


def total_variation_loss(img):
    # img shape==> (1, C, H, W)
    x_diff = img[:, :, :, 1:] - img[:, :, :, :-1]
    y_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
    return torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))


content = load_image(CONTENT_IMAGE_PATH, IMAGE_SIZE)
style   = load_image(STYLE_IMAGE_PATH, IMAGE_SIZE)

if START_FROM_CONTENT:
    combination = content.clone().detach().requires_grad_(True)
else:
    combination = torch.randn_like(content, device=device, requires_grad=True)


vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
for p in vgg.parameters():
    p.requires_grad = False

STYLE_LAYERS = ['0','5','10','19','28']
CONTENT_LAYER = '21'

def extract_features(x, model, layers):
    """Return dict of activations for requested layer indices."""
    features = {}
    out = normalize_batch(x)
    for name, layer in model._modules.items():
        out = layer(out)
        if name in layers:
            features[name] = out
    return features


optimizer = optim.Adam([combination], lr=0.02)   


print("Starting style transfer... (this may take a while)")


content_features = extract_features(content, vgg, [CONTENT_LAYER])
style_features = extract_features(style, vgg, STYLE_LAYERS)


style_grams = {layer: gram_matrix(style_features[layer]) for layer in STYLE_LAYERS}

for i in trange(ITERATIONS, desc="Style Transfer"):
    optimizer.zero_grad()

    combo_feats = extract_features(combination, vgg, STYLE_LAYERS + [CONTENT_LAYER])

    
    c_loss = F.mse_loss(combo_feats[CONTENT_LAYER], content_features[CONTENT_LAYER]) * CONTENT_WEIGHT

   
    s_loss = 0.0
    for layer in STYLE_LAYERS:
        gm_comb = gram_matrix(combo_feats[layer])
        gm_style = style_grams[layer]
        s_loss += F.mse_loss(gm_comb, gm_style)
    s_loss = (s_loss / len(STYLE_LAYERS)) * STYLE_WEIGHT


    tv_loss = TOTAL_VARIATION_WEIGHT * total_variation_loss(combination)

    loss = c_loss + s_loss + tv_loss
    loss.backward()
    optimizer.step()


    with torch.no_grad():
        combination.clamp_(0.0, 1.0)


    if (i == ITERATIONS-1):
        inter_path = os.path.join(INTERMEDIATE_DIR, f"iter_{i+1:04d}.png")
        save_image(combination, inter_path)
        print(f"[iter {i+1:03d}] loss={loss.item():.2f} -> saved {inter_path}")

def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')


save_image(combination, OUTPUT_PATH)
make_combined(content, style, combination, COMBINED_PATH)
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
imshow(content,title='Content Image')
plt.subplot(1,3,2)
imshow(style, title='Style Image')
plt.subplot(1,3,3)
imshow(combination, title='Output Image')
plt.show()

cv.imshow("Output", cv.imread(OUTPUT_PATH))
cv.waitKey(0)
print("Finished. Output saved to:", OUTPUT_PATH, COMBINED_PATH)
