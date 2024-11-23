# AE/ME 7785 IRR. Pratheek Manjunath and Chris Meier. Lab 6 Part 1. 
# Script to build the sign classifier using PyTorch based CNN based
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Constants
IMAGE_SIZE = (64, 64)  # Resize images to 64x64
DATA_FOLDER = "/home/pratheek/Downloads/Curated"
RANDOM_SEED = 747
BATCH_SIZE = 16
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_LABELS = {
    0: "empty wall",
    1: "left",
    2: "right",
    3: "do not enter",
    4: "stop",
    5: "goal"
}

# CNN Model definition with 5 layers
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second Convolutional Block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third Convolutional Block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Fourth Convolutional Block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Fifth Convolutional Block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # After the last pooling layer, calculate the output size of the feature map
        self._conv_output_size = self._get_conv_output_size(IMAGE_SIZE)
        
        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._conv_output_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 6)  # 6 classes
        )
    
    def _get_conv_output_size(self, input_size):
        # Assuming square input (height == width)
        h, w = input_size  # Unpack the height and width
        for _ in range(5):  # 5 pooling layers (one per conv block)
            h = (h - 2) // 2 + 1  # MaxPool2d(2, 2) halves the size
            w = (w - 2) // 2 + 1  # MaxPool2d(2, 2) halves the size
        return h * w * 512  # Output size times the number of channels

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load and preprocess images
def load_images_and_labels(data_folder):
    images = []
    labels = []
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalization for RGB
    ])
    
    for label_str in os.listdir(data_folder):
        label_path = os.path.join(data_folder, label_str)
        if os.path.isdir(label_path):
            label = int(label_str)
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                try:
                    image = Image.open(file_path).convert("RGB")
                    image = transform(image)
                    images.append(image)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return torch.stack(images), torch.tensor(labels)

def main():
    print("Loading images...")
    X, y = load_images_and_labels(DATA_FOLDER)
    print(f"Loaded {len(y)} images.")

    # Split dataset
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model, optimizer, and loss function
    model = CNNModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=CLASS_LABELS.values()))
    # Generate and display the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(CLASS_LABELS.values()))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Save the trained model
    torch.save(model.state_dict(), "vision33_classifier.pth")
    print("Model saved as 'vision33_classifier.pth'.")

if __name__ == "__main__":
    main()