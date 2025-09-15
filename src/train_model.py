# src/train_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_preprocessing import trainloader, testloader, train_data

# -------------------
# Device
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Model
# -------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 37 * 37, 512)
        self.fc2 = nn.Linear(512, 6)  # 6 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net().to(device)

# -------------------
# Loss and Optimizer
# -------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# -------------------
# Training Loop
# -------------------
for epoch in range(10):
    net.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}, Val Acc: {acc:.2f}%")

# -------------------
# Save model
# -------------------
torch.save(net.state_dict(), "intel_model.pth")
print("Model saved as intel_model.pth")
