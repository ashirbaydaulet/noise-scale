import torch
import torch.nn as nn
from src.models.resnet import ResNet18CIFAR

class FusionNet(nn.Module):
    def __init__(self, num_classes=10):
        super(FusionNet, self).__init__()
        
        # EYE 1: The Image Processor (Your ResNet)
        # We remove the final layer because we want features, not decisions yet.
        self.cnn = ResNet18CIFAR(num_classes=num_classes)
        self.cnn.model.fc = nn.Identity() # Remove the last layer
        
        # EYE 2: The Metadata Processor (Simple Neural Network)
        # Takes a "One-Hot" vector (size 10) representing the text label
        self.mlp = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # THE BRAIN: The Fusion Layer
        # ResNet outputs 512 features. MLP outputs 64.
        # We combine them (512 + 64 = 576)
        self.classifier = nn.Linear(512 + 64, num_classes)
        
    def forward(self, image, metadata):
        # 1. Process Image
        img_features = self.cnn(image)
        
        # 2. Process Metadata
        meta_features = self.mlp(metadata)
        
        # 3. Concatenate (Fuse them together)
        combined = torch.cat((img_features, meta_features), dim=1)
        
        # 4. Final Decision
        output = self.classifier(combined)
        return output
