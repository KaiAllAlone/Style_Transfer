import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
from tqdm import tqdm
import os
from datetime import datetime

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Image Processing --------------------
def image_loader(image_name, imsize=512):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')

# -------------------- Loss Modules --------------------
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1,1,1)
        self.std = std.clone().detach().view(-1,1,1)
    def forward(self, img):
        return (img - self.mean) / self.std

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = 0
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = 0
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# -------------------- Build Model --------------------
content_layers_default = ['conv_4']
style_layers_default = ['conv_1','conv_2','conv_3','conv_4','conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
        model.add_module(name, layer)
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)
    # Trim layers after last content/style loss
    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i+1)]
    return model, style_losses, content_losses

# -------------------- Style Transfer --------------------
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img,
                       num_steps=200,style_weight=1e7, content_weight=1):
    print("Building the style transfer model...")
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img)

    optimizer = optim.LBFGS([input_img.requires_grad_()])
    print("Optimizing...")

    step = [0]
    pbar = tqdm(total=num_steps, desc="Style Transfer Progress")

    def closure():
        input_img.data.clamp_(0,1)
        optimizer.zero_grad()
        model(input_img)
        style_score = sum(sl.loss for sl in style_losses) * style_weight
        content_score = sum(cl.loss for cl in content_losses) * content_weight
        loss = style_score + content_score
        loss.backward()
        step[0] += 1
        pbar.n = min(step[0], num_steps)
        pbar.refresh()
        return loss

    # LBFGS requires multiple calls; stop after num_steps
    while step[0] < num_steps:
        optimizer.step(closure)

    pbar.close()
    input_img.data.clamp_(0,1)
    return input_img

# -------------------- Main --------------------
def main(style_img_path, content_img_path, output_dir='outputs',
         imsize=512,num_steps=200,style_weight=1e5,content_weight=1):
    
    os.makedirs(output_dir, exist_ok=True)

    style_img = image_loader(style_img_path, imsize)
    content_img = image_loader(content_img_path, imsize)
    assert style_img.size() == content_img.size(), "Style and Content images must be same size"

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485,0.456,0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229,0.224,0.225]).to(device)

    input_img = content_img.clone()

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img,
                                num_steps=num_steps, style_weight=style_weight,
                                content_weight=content_weight)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    imshow(content_img, title='Content Image')
    plt.subplot(1,3,2)
    imshow(style_img, title='Style Image')
    plt.subplot(1,3,3)
    imshow(output, title='Output Image')
    plt.show()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f'styled_output_{timestamp}.jpg')
    unloader = transforms.ToPILImage()
    image_to_save = output.cpu().clone().squeeze(0)
    image_to_save = unloader(image_to_save)
    image_to_save.save(output_filename)
    print(f"Saved styled image as {output_filename}")

# -------------------- Run Example --------------------
if __name__ == "__main__":
    style_img_path = r"Style_Transfer\Aarohan2k25\8-bit.jpg"
    content_img_path = r"Style_Transfer\Aarohan2k25\buildings.png"
    main(style_img_path,content_img_path,imsize=512)
