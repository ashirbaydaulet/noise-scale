import torch
from torch.utils.data import Dataset
from PIL import Image

class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (np.array): Shape (N, 32, 32, 3) - Raw numpy images
            labels (list or np.array): Class labels
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Get image and label
        img = self.images[idx]
        label = self.labels[idx]

        # 2. Convert Numpy -> PIL Image (Required for most transforms like RandomCrop)
        # CIFAR images are (H, W, C) in numpy, which PIL likes.
        img = Image.fromarray(img)

        # 3. Apply Transforms (Flip, Crop, ToTensor, Normalize)
        if self.transform:
            img = self.transform(img)

        return img, label
