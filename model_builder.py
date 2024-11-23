# AE/ME 7785 IRR. Pratheek Manjunath and Chris Meier. Lab 6 Part 1. 
# Script to build the sign classifier using ResNet-18.
# Half the layers use pretrained weights from ImageNet. The unfrozen layers are retrained with our dataset.
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Constants
DATA_FOLDER = "/home/pratheek/Downloads/Curated"
MODEL_PATH = "resnet18_classifier.pth"  # File to save the trained model
IMAGE_SIZE = (224, 224)  # ResNet input size
BATCH_SIZE = 32
EPOCHS = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class labels (adjust to your own classes)
CLASS_LABELS = {
    0: "empty wall",
    1: "left",
    2: "right",
    3: "do not enter",
    4: "stop",
    5: "goal"
}

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        for label_str in os.listdir(data_folder):
            label_path = os.path.join(data_folder, label_str)
            if os.path.isdir(label_path):
                label = int(label_str)
                for file_name in os.listdir(label_path):
                    self.image_paths.append(os.path.join(label_path, file_name))
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
full_dataset = CustomDataset(DATA_FOLDER, transform)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load pre-trained ResNet-18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features

for param in model.parameters():        # Freeze earlier layers and unfreeze later layers
    param.requires_grad = False
for param in list(model.parameters())[-10:]:  # Unfreeze the last 10 layers for retraining
    param.requires_grad = True

# Add dropout and then update the fully connected layer
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),  # 30% dropout
    nn.Linear(512, 6)  # Assuming 6 classes
)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# Training loop
def train():
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Training Loss: {running_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved as '{MODEL_PATH}'.")

# Evaluation loop
def evaluate():
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    # Print classification report and confusion matrix
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(CLASS_LABELS.values())))
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(CLASS_LABELS.values()))
    disp.plot(cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.title("Confusion Matrix (Post-build evaluation with 80-20 data split)")
    plt.show()

if __name__ == "__main__":
    train()
    evaluate()