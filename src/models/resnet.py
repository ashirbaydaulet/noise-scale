import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18CIFAR, self).__init__()
        
        # 1. Load the standard ResNet-18 structure
        # We don't use 'pretrained=True' because we are modifying the architecture
        # and training from scratch on a different image size.
        self.model = resnet18(weights=None)
        
        # 2. CRITICAL CHANGE FOR CIFAR-10 (32x32 images)
        # Standard ResNet uses a 7x7 conv with stride 2 and maxpool.
        # This shrinks 32x32 images to 8x8 too quickly. 
        # We replace it with a 3x3 conv with stride 1 and REMOVE maxpool.
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity() # Effectively removes the layer
        
        # 3. Change the final classification layer
        # The standard model outputs 1000 classes (ImageNet). We need 10 (CIFAR).
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
