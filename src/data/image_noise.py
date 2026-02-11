import torch
import torchvision.transforms as T

def add_gaussian_noise(image_tensor, severity_level):
    """
    Adds Gaussian noise (Static).
    Severity: Standard Deviation (0.0 to 1.0)
    """
    noise = torch.randn_like(image_tensor) * severity_level
    noisy_image = image_tensor + noise
    return torch.clamp(noisy_image, 0.0, 1.0)

def add_blur_noise(image_tensor, severity_level):
    """
    Adds Gaussian Blur (Smearing).
    Severity: Sigma (Standard Deviation of the Gaussian Kernel).
              Reasonable range: 0.1 to 3.0
    """
    # 1. Safety check (Sigma must be > 0)
    if severity_level <= 0.0:
        return image_tensor
        
    # 2. Kernel size must be odd. We lock it at 5x5.
    #    We only vary sigma (intensity).
    k_size = 5
    
    # 3. Create the blurrer
    blurrer = T.GaussianBlur(kernel_size=k_size, sigma=severity_level)
    
    return blurrer(image_tensor)
