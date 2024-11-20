# AE/ME 7785 IRR. Pratheek Manjunath and Chris Meier. Lab 6 Part 1. Script to test the classification model built using Scikit-learn's MLP algorithm.
import os
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import matplotlib.pyplot as plt

# Constants
MODEL_PATH = "vision33_classifier.pkl"     # This model should already be in the current directory
IMAGE_SIZE = (64, 64)                      # Matches image size used for training
DATA_FOLDER = "/home/pratheek/Downloads/test" # Replace with the path to the test dataset. Must contain subfolders named 0, 1 ,2 , 3, 4, and 5.
CLASS_LABELS = {
    0: "empty wall",
    1: "left",
    2: "right",
    3: "do not enter",
    4: "stop",
    5: "goal"
}

# Load the scaler used to fit the training data
def load_scaler():
    scaler = joblib.load('scaler33.pkl')  
    return scaler

# Preprocess a single image to match the format used during training: Resize, Flatten, and Normalize
def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image).flatten()
        image_array = image_array / 255.0  # Normalize to [0, 1]
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Load test images from the specified folder
def load_images_from_folder(folder_path):
    images = []
    labels = []
    for label_str in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_str)
        if os.path.isdir(label_path):
            label = int(label_str)  # Convert folder name to integer label
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                image_data = preprocess_image(file_path)
                if image_data is not None:
                    images.append(image_data)
                    labels.append(label)
    return np.array(images), np.array(labels)

def test_model(model_path, dataset_folder):
    # Load the trained model
    print("Loading trained model...")
    model = joblib.load(model_path)

    # Load the scaler used during training
    scaler = load_scaler()

    # Load the test dataset
    print(f"Loading dataset from {dataset_folder}...")
    X, y = load_images_from_folder(dataset_folder)
    if len(X) == 0:
        print("No valid images found in the dataset folder.")
        return

    X = scaler.transform(X)
    if np.any(np.isnan(X)):
        print("Warning: Test data contains NaN values after scaling. Check input data.")

    # Predict the classes
    print("Making predictions...")
    y_pred = model.predict(X)
    print("\nPredictions:")
    for i, pred in enumerate(y_pred):
        print(f"Image {i + 1}: Predicted Label = {pred} ({CLASS_LABELS[pred]})")

    # Evaluate accuracy
    print("\nEvaluating accuracy...")
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=CLASS_LABELS.values(), zero_division=0))

    # Generate and display the confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y, y_pred, labels=list(CLASS_LABELS.keys()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS.values())
    disp.plot(cmap=plt.cm.Oranges) 
    plt.tight_layout()
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    test_model(MODEL_PATH, DATA_FOLDER)