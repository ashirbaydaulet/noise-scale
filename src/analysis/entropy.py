
import torch
import torch.nn.functional as F

def calculate_entropy(logits):
    """
    Calculates the uncertainty (Entropy) of the model's predictions.
    Args:
        logits (Tensor): Raw output from the model (before Softmax).
    Returns:
        entropies (Tensor): A single value per image representing uncertainty.
    """
    # 1. Convert logits to probabilities (0.0 to 1.0)
    probs = F.softmax(logits, dim=1)
    
    # 2. Calculate Log Probabilities
    log_probs = F.log_softmax(logits, dim=1)
    
    # 3. Entropy Formula: -Sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=1)
    
    return entropy
