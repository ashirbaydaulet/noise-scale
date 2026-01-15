import torch

def add_gaussian_noise(image_tensor, severity_level):
    """
    Adds Gaussian noise to an image tensor.
    
    Args:
        image_tensor (torch.Tensor): The clean image with values [0, 1].
        severity_level (float): The standard deviation (sigma) of the noise.
                                0.0 means no noise. 1.0 is severe noise.
    
    Returns:
        torch.Tensor: The noisy image, clipped to range [0, 1].
    """
    # 1. Generate random noise with the same shape as the image
    #    torch.randn_like generates numbers from a normal distribution (bell curve)
    noise = torch.randn_like(image_tensor) * severity_level
    
    # 2. Add noise to the clean image
    noisy_image = image_tensor + noise
    
    # 3. Clip values! (CRITICAL STEP)
    #    We must ensure pixel values don't go below 0.0 or above 1.0
    #    or the model (and plotting tools) will break.
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
    
    return noisy_image
