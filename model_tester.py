# AE/ME 7785 IRR. Pratheek Manjunath and Chris Meier. Lab 6 Part 1. Script to test the classification model built using PyTorch and ResNet18.
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Constants
MODEL_PATH = "resnet18_classifier.pth"  # Path to your saved model
TEST_DATA_FOLDER = "/home/pratheek/Downloads/test-CNN"  # Provide path to 2024F_G. Must contain subfolders 0 through 5.
IMAGE_SIZE = (224, 224)  # Resize images to match model input
CLASS_LABELS = {
    0: "empty wall",
    1: "left",
    2: "right",
    3: "do not enter",
    4: "stop",
    5: "goal"
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image preprocessing (same as during training)
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    try:
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Load test dataset using provided function
def load_images_from_folder(folder_path):
    images, labels = [], []
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
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    # Match the training architecture
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(CLASS_LABELS))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Load test dataset
    print("Loading test dataset...")
    images, labels = load_images_from_folder(TEST_DATA_FOLDER)
    if not images:
        print("No images found in the test dataset.")
        return

    predicted, true_labels = [], labels.numpy()

    # Evaluate each image individually
    with torch.no_grad():
        for image in images:
            image = image.to(DEVICE)
            output = model(image)
            _, pred = torch.max(output, 1)
            predicted.append(pred.item())

    predicted = np.array(predicted)

    # Calculate accuracy
    accuracy = (predicted == true_labels).sum() / len(labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted, target_names=CLASS_LABELS.values(), zero_division=0))

    # Display confusion matrix
    cm = confusion_matrix(true_labels, predicted, labels=list(CLASS_LABELS.keys()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS.values())
    disp.plot(cmap=plt.cm.Oranges)
    plt.tight_layout()
    plt.title("Confusion Matrix (Test data)")
    plt.show()

if __name__ == "__main__":
    test_model()