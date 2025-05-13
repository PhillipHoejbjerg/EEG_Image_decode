import lpips
import numpy as np


import torch
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
from torch.nn import functional as F


from PIL import Image

def calculate_inception_score(img: Image.Image, inception_model, device):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = inception_model(img_tensor)
        pred = F.softmax(pred, dim=1).cpu().numpy()

    # IS for batch size 1: KL(p(y|x) || p(y)) where p(y) ≈ p(y|x)
    py = np.mean(pred, axis=0)
    scores = pred * (np.log(pred + 1e-10) - np.log(py + 1e-10))
    kl = np.sum(scores)
    inception_score = np.exp(kl)
    return inception_score


def compute_lpips(img1, img2, lpips_model, device):
    img1_tensor = transforms.ToTensor()(img1).unsqueeze(0).to(device)
    img2_tensor = transforms.ToTensor()(img2).unsqueeze(0).to(device)
    return lpips_model(img1_tensor, img2_tensor).item()


from skimage.metrics import structural_similarity as ssim

def compute_ssim(img1, img2):
    img1 = img1.resize((256, 256)).convert('L')
    img2 = img2.resize((256, 256)).convert('L')
    return ssim(np.array(img1), np.array(img2))


def color_histogram_distance(img1, img2):
    hist1 = np.histogram(img1.convert('RGB'), bins=256, range=(0, 255))[0]
    hist2 = np.histogram(img2.convert('RGB'), bins=256, range=(0, 255))[0]
    return np.linalg.norm(hist1 - hist2)