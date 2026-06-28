import os

# Data and directory settings
DATA_DIR = "Data"
CATEGORIES = ["ok", "peace", "stop", "like", "dislike"]
IMG_SIZE = 64  # kept for reference, no longer used for pixel features

# MediaPipe hand landmark model
LANDMARK_MODEL_PATH = "hand_landmarker.task"

# Model and saved file settings
MODEL_FILENAME = "hand_gesture_sklearn.joblib"

if __name__ == "__main__":
    print("=== Project Settings ===")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Class Categories: {CATEGORIES}")
    print(f"Image Dimensions: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Saved Model Filename: {MODEL_FILENAME}")
