import numpy as np
import joblib
from data_loader import load_data
from model import create_model
from settings import MODEL_FILENAME

def run_training():
    print("Loading training data (Train) via Data Loader...")
    X_train, y_train = load_data("train")

    if len(X_train) == 0:
        print("Error: No training images found. Check your 'Data/train' directory path.")
        return

    print(f"Training data shape: {X_train.shape}")

    print("\nStarting model training process...")
    model = create_model()
    model.fit(X_train, y_train)
    print("Training completed successfully.")

    joblib.dump(model, MODEL_FILENAME)
    print(f"Final model exported and saved to: {MODEL_FILENAME}")

if __name__ == "__main__":
    print("=== Starting Training Phase ===")
    run_training()
