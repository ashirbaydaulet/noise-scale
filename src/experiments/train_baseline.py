import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import sys
import os
import numpy as np

# Fix path to find 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.cifar_loader import load_cifar_batch
from src.data.cifar_dataset import CIFAR10Dataset
from src.models.resnet import ResNet18CIFAR

def train():
    # Configuration
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01 
    EPOCHS = 20
    DATA_PATH = '/Users/daulet/Desktop/data centric ai/cifar-10-batches-py'
    SAVE_PATH = os.path.join('src', 'experiments', 'saved_models', 'baseline_model.pth')

    # Define Transforms (The "Upgrade")
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    print("Loading ALL data batches for maximum accuracy...")
    all_images = []
    all_labels = []
    for i in range(1, 6):
        fpath = os.path.join(DATA_PATH, f'data_batch_{i}')
        if os.path.exists(fpath):
            imgs, lbls = load_cifar_batch(fpath)
            all_images.append(imgs)
            all_labels.extend(lbls)
        else:
            print(f"Warning: Could not find {fpath}")

    x_train = np.concatenate(all_images)
    y_train = np.array(all_labels)

    # Create Dataset
    train_dataset = CIFAR10Dataset(x_train, y_train, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Setup Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device} | Images: {len(x_train)}")
    
    model = ResNet18CIFAR(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    # SGD is often better than Adam for ResNet/CIFAR to reach high accuracy
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Loop
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        scheduler.step()
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

    # Save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"SUCCESS: Saved to {SAVE_PATH}")

if __name__ == "__main__":
    train()
