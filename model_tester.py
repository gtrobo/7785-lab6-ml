# AE/ME 7785 IRR. Pratheek Manjunath and Chris Meier. Lab 6 Part 1. Script to test the classification model built using Scikit-learn's MLP algorithm.
import os
import numpy as np
from PIL import Image
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Constants
MODEL_PATH = "vision33_classifier.pkl"  # Path to your saved model
IMAGE_SIZE = (64, 64)  # Resize images to match the training size
CLASS_LABELS = {
    0: "empty wall",
    1: "left",
    2: "right",
    3: "do not enter",
    4: "stop",
    5: "goal"
}

def preprocess_image(image_path):
    # Preprocess a single image to match the format used during training: Resize, Flatten, and Normalize.
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image).flatten()
        image_array = image_array / 255.0  # Normalize to [0, 1]
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def load_images_from_folder(folder_path):
    # Load all images from a folder and their corresponding labels if available.
    images = []
    labels = []
    for label_str in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_str)
        if os.path.isdir(label_path):
            label = int(label_str)
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                image_data = preprocess_image(file_path)
                if image_data is not None:
                    images.append(image_data)
                    labels.append(label)
    return np.array(images), np.array(labels)

def test_model(dataset_folder, model_path):
    """
    Load the model and test it on a dataset.
    """
    # Load the model
    print("Loading model...")
    model = joblib.load(model_path)

    # Load the dataset
    print(f"Loading dataset from {dataset_folder}...")
    X, y = load_images_from_folder(dataset_folder)
    if len(X) == 0:
        print("No valid images found in the dataset folder.")
        return

    # Predict the classes
    print("Making predictions...")
    predictions = model.predict(X)

    # Print predictions
    print("\nPredictions:")
    for i, pred in enumerate(predictions):
        print(f"Image {i + 1}: Predicted Label = {pred} ({CLASS_LABELS[pred]})")

    # Evaluate accuracy if ground truth labels are available
    if len(y) > 0:
        print("\nEvaluating accuracy...")
        accuracy = accuracy_score(y, predictions)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:\n", classification_report(y, predictions, target_names=CLASS_LABELS.values()))

if __name__ == "__main__":
    dataset_folder = "/home/pratheek/Downloads/Curated"  # Replace with the path to the test dataset. Must contain subfolders named 0, 1 ,2 , 3, 4, and 5.
    test_model(dataset_folder, MODEL_PATH)