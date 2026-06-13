import os, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from PIL import Image

DATA_ROOT = "data/"   # contains train/ and test/

print("Dataset information: ")
# Base transform (no augmentation) — used for test
base_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

train_dataset = datasets.ImageFolder(DATA_ROOT + "train", transform=base_tf)
test_dataset  = datasets.ImageFolder(DATA_ROOT + "test",  transform=base_tf)

print("Train: ", len(train_dataset))
print("Test: ", len(test_dataset))
print("Total: ", len(train_dataset) + len(test_dataset))

print(train_dataset.class_to_idx)
# Verify counts
from collections import Counter
counts = Counter(train_dataset.targets)
print({train_dataset.classes[k]: v for k, v in counts.items()})


print("\nImages information: ")
img_path = "data/train/angry/Training_3908.jpg"
with Image.open(img_path) as img:
    print("Size: ", img.size)
    print("Mode: ", img.mode) # L: Grayscale (1 channel)
    print("Format: ", img.format)