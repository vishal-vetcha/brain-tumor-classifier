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
        self.classes = sorted(os.listdir(root_dir)) # Get class names

        for label_idx, label in enumerate(self.classes):
            label_path = os.path.join(root_dir, label)
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                try:
                    img = cv2.imread(file_path) # Read image
                    if img is None:
                        print(f"Warning: Could not open image at {file_path}. Skipping.")
                        continue
                    self.images.append(file_path)
                    self.labels.append(label_idx) # Assign numerical label
                except Exception as e:
                    print(f"Error processing {file_path}: {e}. Skipping.")
                    continue

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
            img = cv2.resize(img, (224, 224)) # Resize image
            img = Image.fromarray(img) # To PIL Image
            if self.transform:
                img = self.transform(img) # Apply transforms
            return img, label
        except Exception as e:
            print(f"Error loading/processing {img_path} in __getitem__: {e}")
            dummy_img = torch.zeros((3, 224, 224))
            dummy_label = -1
            return dummy_img, dummy_label