# src/test_visualize.py
import torch
import random
import matplotlib.pyplot as plt
from data_preprocessing import test_data, train_data
from train_model import Net

# -------------------
# Device
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Load Model
# -------------------
net = Net().to(device)
net.load_state_dict(torch.load("intel_model.pth"))
net.eval()

# -------------------
# Pick one random image per class
# -------------------
class_to_indices = {i: [] for i in range(len(train_data.classes))}
for idx, (_, label) in enumerate(test_data):
    class_to_indices[label].append(idx)

images, labels = [], []
for cls, indices in class_to_indices.items():
    idx = random.choice(indices)
    img, label = test_data[idx]
    images.append(img)
    labels.append(label)

images = torch.stack(images).to(device)
labels = torch.tensor(labels).to(device)

# -------------------
# Predictions
# -------------------
with torch.no_grad():
    outputs = net(images)
    _, preds = torch.max(outputs, 1)

# -------------------
# Visualization
# -------------------
def imshow_grid(images, labels, preds, classes):
    num = len(images)
    plt.figure(figsize=(12, 6))
    for i in range(num):
        img = images[i].cpu().numpy().transpose((1,2,0))
        img = img * 0.5 + 0.5
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.title(f"P: {classes[preds[i]]}\nT: {classes[labels[i]]}", fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

imshow_grid(images, labels, preds, train_data.classes)
