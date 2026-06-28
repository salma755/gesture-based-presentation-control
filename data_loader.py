import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from settings import DATA_DIR, CATEGORIES, LANDMARK_MODEL_PATH

_landmarker = None

def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        base_opts = mp_python.BaseOptions(model_asset_path=LANDMARK_MODEL_PATH)
        opts = vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            running_mode=vision.RunningMode.IMAGE,
        )
        _landmarker = vision.HandLandmarker.create_from_options(opts)
    return _landmarker

def extract_landmarks(img_path=None, mp_image=None):
    """Return normalized 63-dim landmark vector, or None if hand not detected."""
    if mp_image is None:
        mp_image = mp.Image.create_from_file(img_path)
    result = _get_landmarker().detect(mp_image)
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    # Normalize: center on wrist, scale by hand span
    origin = coords[0].copy()
    coords -= origin
    scale = np.linalg.norm(coords[9])  # wrist-to-middle-MCP distance
    if scale > 0:
        coords /= scale
    return coords.flatten()

def load_data(split_name):
    X, y = [], []
    split_path = os.path.join(DATA_DIR, split_name)

    if not os.path.exists(split_path):
        print(f"Error: Directory {split_path} does not exist!")
        return np.array(X), np.array(y)

    skipped = 0
    for category in CATEGORIES:
        category_path = os.path.join(split_path, category)
        class_num = CATEGORIES.index(category)
        if not os.path.exists(category_path):
            continue
        for img_name in os.listdir(category_path):
            try:
                features = extract_landmarks(img_path=os.path.join(category_path, img_name))
                if features is None:
                    skipped += 1
                    continue
                X.append(features)
                y.append(class_num)
            except Exception:
                skipped += 1

    if skipped:
        print(f"  (skipped {skipped} images where no hand was detected)")
    return np.array(X, dtype=np.float32), np.array(y)

if __name__ == "__main__":
    print("Testing Landmark Data Loader...")
    X_sample, y_sample = load_data("train")
    print(f"Success: Loaded training sample with shape: {X_sample.shape}")
