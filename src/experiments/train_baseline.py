import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Fix path to import from 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.cifar_loader import load_cifar_batch
from src.models.resnet import ResNet18CIFAR

def train():
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 5
    # Your specific path
    DATA_PATH = '/Users/daulet/Desktop/data centric ai/cifar-10-batches-py'

    print("Loading data...")
    batch_file = os.path.join(DATA_PATH, 'data_batch_1')
    
    if not os.path.exists(batch_file):
        print(f"ERROR: File not found at {batch_file}")
        return

    images, labels = load_cifar_batch(batch_file)

    # Transform for PyTorch (N, H, W, C) -> (N, C, H, W)
    images = images.transpose(0, 3, 1, 2) 
    x_train = torch.tensor(images, dtype=torch.float32) / 255.0
    y_train = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = ResNet18CIFAR().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(loader):.4f}")

    print("Finished Training")
    
    # Save model
    os.makedirs('src/experiments/saved_models', exist_ok=True)
    torch.save(model.state_dict(), 'src/experiments/saved_models/baseline_model.pth')
    print("Model saved!")

# THIS IS THE PART THAT WAS LIKELY MISSING
if __name__ == "__main__":
    train()
