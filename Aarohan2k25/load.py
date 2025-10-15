import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- Utility ----------
def im_convert(tensor):
    """Convert tensor to displayable image"""
    image = tensor.to("cpu").clone().detach().squeeze()
    image = image.numpy().transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = np.clip(image, 0, 1)
    return image

# ---------- Main Function ----------
def apply_saved_style(content_path, saved_model_path="Programs\Aarohan2k25\style_transfer_model.pth", device='cuda'):
    """
    Load a saved StyleTransferModel and apply the stylization.
    Returns: PIL.Image and the tensor.
    """
    if not os.path.exists(saved_model_path):
        raise FileNotFoundError(f"Saved model not found: {saved_model_path}")

    # Load the saved model
    model = torch.load(saved_model_path, map_location=device)
    model.to(device)

    # Access the target tensor (already optimized)
    stylized_tensor = model.target

    # Convert to displayable image
    stylized_image = im_convert(stylized_tensor)

    # Optionally display
    plt.figure(figsize=(6, 6))
    plt.imshow(stylized_image)
    plt.axis('off')
    plt.title("Stylized Output")
    plt.show()

    # Convert to PIL for saving
    pil_image = Image.fromarray((stylized_image * 255).astype(np.uint8))
    return pil_image, stylized_tensor

# ---------- Example Usage ----------
content_path = r'Programs\Aarohan2k25\WIN_20251015_15_02_39_Pro.jpg'
pil_img, tensor_img = apply_saved_style(content_path, saved_model_path="Programs\Aarohan2k25\style_transfer_model.pth")

# Save the stylized output
pil_img.save("reused_stylized_output.jpg")
print("✅ Stylized image saved to reused_stylized_output.jpg")
