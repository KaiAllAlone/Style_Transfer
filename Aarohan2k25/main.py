# backend/style_transfer_full.py
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
from tqdm import trange
import os

def run_style_transfer(content_path, style_path, output_path,
                       image_size=500, iterations=200,
                       content_weight=1.0, style_weight=1e5, tv_weight=1e-6,
                       start_from_content=True):
    """Runs neural style transfer on given images and saves output to output_path."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running style transfer on: {device}")

    # -----------------------------
    # Helper functions
    # -----------------------------
    def load_image(path, size):
        img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
        tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
        return tensor

    def save_image(tensor, path):
        img = tensor.detach().cpu().clamp(0, 1).squeeze(0)
        pil = transforms.ToPILImage()(img)
        pil.save(path)

    def normalize_batch(batch):
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        return (batch - mean[None, :, None, None]) / std[None, :, None, None]

    def gram_matrix(input):
        b, c, h, w = input.size()
        features = input.view(b * c, h * w)
        G = torch.mm(features, features.t())
        return G / (c * h * w)

    def total_variation_loss(img):
        x_diff = img[:, :, :, 1:] - img[:, :, :, :-1]
        y_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
        return torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))

    # -----------------------------
    # Load images
    # -----------------------------
    content = load_image(content_path, image_size)
    style = load_image(style_path, image_size)

    combination = content.clone().detach().requires_grad_(True) if start_from_content else \
                  torch.randn_like(content, device=device, requires_grad=True)

    # -----------------------------
    # Load VGG19
    # -----------------------------
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    for p in vgg.parameters():
        p.requires_grad = False

    STYLE_LAYERS = ['0','5','10','19','28']
    CONTENT_LAYER = '21'

    def extract_features(x, model, layers):
        feats = {}
        out = normalize_batch(x)
        for name, layer in model._modules.items():
            out = layer(out)
            if name in layers:
                feats[name] = out
        return feats

    # -----------------------------
    # Precompute features
    # -----------------------------
    content_feats = extract_features(content, vgg, [CONTENT_LAYER])
    style_feats = extract_features(style, vgg, STYLE_LAYERS)
    style_grams = {l: gram_matrix(style_feats[l]) for l in STYLE_LAYERS}

    optimizer = optim.Adam([combination], lr=0.02)

    # -----------------------------
    # Training loop
    # -----------------------------
    for i in trange(iterations, desc="Style Transfer"):
        optimizer.zero_grad()

        combo_feats = extract_features(combination, vgg, STYLE_LAYERS + [CONTENT_LAYER])

        c_loss = F.mse_loss(combo_feats[CONTENT_LAYER], content_feats[CONTENT_LAYER]) * content_weight

        s_loss = sum(
            F.mse_loss(gram_matrix(combo_feats[l]), style_grams[l]) for l in STYLE_LAYERS
        ) / len(STYLE_LAYERS) * style_weight

        tv_loss = tv_weight * total_variation_loss(combination)

        loss = c_loss + s_loss + tv_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            combination.clamp_(0.0, 1.0)

    save_image(combination, output_path)
    print(f"✅ Style transfer complete: {output_path}")
    return output_path