import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        for label_idx, label in enumerate(self.classes):
            label_path = os.path.join(root_dir, label)
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                self.images.append(file_path)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Open image using OpenCV
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img = cv2.resize(img, (224, 224))

        # Convert to PIL Image (for transforms)
        img = Image.fromarray(img)

        # Apply transforms if any
        if self.transform:
            img = self.transform(img)

        return img, label
