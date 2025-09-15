# src/data_preprocessing.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# -------------------
# Dataset Paths
# -------------------
train_dir = "intel_data/seg_train/seg_train"
test_dir  = "intel_data/seg_test/seg_test"

# -------------------
# Transforms
# -------------------
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# -------------------
# Load Dataset
# -------------------
train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data  = datasets.ImageFolder(test_dir, transform=transform)

# -------------------
# DataLoaders
# -------------------
trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
testloader  = DataLoader(test_data, batch_size=32, shuffle=False)

# -------------------
# Print classes
# -------------------
print("Classes:", train_data.classes)
