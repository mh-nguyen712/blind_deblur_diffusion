import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import re
from torch import Tensor
import torch
import numpy as np

class LabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the classes as subdirectories.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        if transform is None:
            # Define transformation
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to [-1, 1]
            ])
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Scan root_dir for classes and images
        for class_dir in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_dir)
            
            # Extract the integer label from the folder name
            match = re.match(r'class_(\d+)', class_dir)
            if match:
                label = int(match.group(1))  # Extract integer value from 'class_i'
                
                # Add images and labels
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")  # Convert to RGB in case of grayscale images
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label



def stable_division(a: Tensor, b: Tensor, epsilon: float = 1e-7):
    b = torch.where(
        b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign()
    )
    return a / b


def to_zeros_and_ones(a: Tensor):
    if a.ndim == 3:
        a -= a.min()
        a /= a.max()
    elif a.ndim == 4:
        dim = np.prod(a.shape[1:])
        min_values = torch.min(a.view(-1, dim), dim=-1).values
        a -= min_values[:, None, None, None]
        max_values = torch.max(a.view(-1, dim), dim=-1).values
        a /= max_values[:, None, None, None]
    return a