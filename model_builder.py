# AE/ME 7785 IRR. Pratheek Manjunath and Chris Meier. Lab 6 Part 1. Script to build the sign classifier using Scikit-learn's MLP algorithm.

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from PIL import Image

# Constants
IMAGE_SIZE = (64, 64)  # Resize images
DATA_FOLDER = "/home/pratheek/Downloads/Curated"
RANDOM_STATE = 42
CLASS_LABELS = {
    0: "empty wall",
    1: "left",
    2: "right",
    3: "do not enter",
    4: "stop",
    5: "goal"
}

def load_images_and_labels(data_folder):
    # Load images and their labels from the curated folder
    images = []
    labels = []
    for label_str in os.listdir(data_folder):
        label_path = os.path.join(data_folder, label_str)
        if os.path.isdir(label_path):
            label = int(label_str)  # Folder names should correspond to labels
            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)
                try:
                    image = Image.open(file_path).convert("RGB")
                    image = image.resize(IMAGE_SIZE)
                    images.append(np.array(image).flatten())
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return np.array(images), np.array(labels)

def main():
    # Load dataset
    print("Loading images...")
    X, y = load_images_and_labels(DATA_FOLDER)
    X, y = shuffle(X, y, random_state=RANDOM_STATE)
    print(f"Loaded {len(y)} images.")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # Normalize the data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Create MLP classifier
    print("Training the MLP classifier...")
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=CLASS_LABELS.values()))

    # Generate and display the confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=list(CLASS_LABELS.keys()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS.values())
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Save the model
    import joblib
    joblib.dump(model, "vision33_classifier.pkl")
    print("Model saved as vision33_classifier, after the name of our group")

if __name__ == "__main__":
    main()