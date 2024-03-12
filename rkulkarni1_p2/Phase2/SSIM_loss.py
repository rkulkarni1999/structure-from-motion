import torch
import torch.nn.functional as F

def gaussian_kernel(size: int, sigma: float):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.view(1, -1) * g.view(-1, 1)

def calculate_ssim(img1, img2, size: int = 11, sigma: float = 1.5, C1: float = 0.01**2, C2: float = 0.03**2, H: int = 400, W:int =400):
    # Reshape images from (h*w, 3) to (N, C, H, W)
    img1 = img1.view(-1, 3, H, W)  # Assuming img1 is a tensor of shape (h*w, 3)
    img2 = img2.view(-1, 3, H, W)  # Assuming img2 is a tensor of shape (h*w, 3)
    
    channel = img1.size(1)
    kernel = gaussian_kernel(size, sigma).to(img1.device)
    kernel = kernel.expand(channel, 1, size, size)
    
    mu1 = F.conv2d(img1, kernel, padding=size//2, groups=channel)
    mu2 = F.conv2d(img2, kernel, padding=size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=size//2, groups=channel) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()