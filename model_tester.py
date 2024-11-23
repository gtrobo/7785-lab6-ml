# AE/ME 7785 IRR. Pratheek Manjunath and Chris Meier. Lab 6 Part 1. Script to test the classification model built using Scikit-learn's MLP algorithm.
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Constants
MODEL_PATH = "vision33_classifier.pth"  # Path to the saved model
TEST_DATA_FOLDER = "/home/pratheek/Downloads/test-CNN"  # Path to your test dataset
IMAGE_SIZE = (64, 64)  # Resize images to 64x64
CLASS_LABELS = {
    0: "empty wall",
    1: "left",
    2: "right",
    3: "do not enter",
    4: "stop",
    5: "goal"
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN Model definition (same as training)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self._conv_output_size = self._get_conv_output_size(IMAGE_SIZE)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._conv_output_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 6)
        )
    
    def _get_conv_output_size(self, input_size):
        h, w = input_size
        for _ in range(5):
            h = (h - 2) // 2 + 1
            w = (w - 2) // 2 + 1
        return h * w * 512

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Image preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Match training normalization
    ])
    try:
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Load test dataset
def load_images_from_folder(folder_path):
    images = []
    labels = []
    for label_str in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_str)
        if os.path.isdir(label_path):
            label = int(label_str)
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                image_tensor = preprocess_image(file_path)
                if image_tensor is not None:
                    images.append(image_tensor)
                    labels.append(label)
    return images, torch.tensor(labels, dtype=torch.long)

# Test the model
def test_model():
    # Load the trained model
    print("Loading the trained model...")
    model = CNNModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Load test dataset
    print("Loading test dataset...")
    images, labels = load_images_from_folder(TEST_DATA_FOLDER)
    if not images:
        print("No images found in the test dataset.")
        return

    predicted = []
    true_labels = labels.cpu().numpy()

    # Test each image individually
    with torch.no_grad():
        for i, image in enumerate(images):
            image = image.to(DEVICE)
            output = model(image)
            _, pred = torch.max(output, 1)
            predicted.append(pred.item())

    predicted = np.array(predicted)

    # Evaluate accuracy
    accuracy = (predicted == true_labels).sum() / len(labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted, target_names=CLASS_LABELS.values(), zero_division=0))

    # Generate and display the confusion matrix
    cm = confusion_matrix(true_labels, predicted, labels=list(CLASS_LABELS.keys()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS.values())
    disp.plot(cmap=plt.cm.Oranges)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    test_model()