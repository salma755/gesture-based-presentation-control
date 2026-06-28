import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from settings import MODEL_FILENAME, CATEGORIES, LANDMARK_MODEL_PATH


def _build_live_landmarker():
    base_opts = mp_python.BaseOptions(model_asset_path=LANDMARK_MODEL_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(opts)


def _extract_live_landmarks(landmarker, frame_rgb, timestamp_ms):
    """Return (landmarks_array, hand_rect, raw_lm_list) or (None, None, None)."""
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect_for_video(mp_img, timestamp_ms)
    if not result.hand_landmarks:
        return None, None, None

    lm = result.hand_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

    # Normalized landmark feature (same as data_loader)
    origin = coords[0].copy()
    coords -= origin
    scale = np.linalg.norm(coords[9])
    if scale > 0:
        coords /= scale
    features = coords.flatten()

    # Bounding box in pixel space for drawing
    h, w = frame_rgb.shape[:2]
    xs = [int(p.x * w) for p in lm]
    ys = [int(p.y * h) for p in lm]
    pad = 20
    rect = (
        max(0, min(xs) - pad),
        max(0, min(ys) - pad),
        min(w, max(xs) + pad),
        min(h, max(ys) + pad),
    )
    return features, rect, lm


def _draw_hand_connections(frame, lm_list):
    """Draw MediaPipe hand skeleton on frame."""
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm_list]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (0, 220, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)


def run_inference():
    if not os.path.exists(MODEL_FILENAME):
        print("Error: Model file not found. Please run train.py first.")
        return
    if not os.path.exists(LANDMARK_MODEL_PATH):
        print(f"Error: '{LANDMARK_MODEL_PATH}' not found. Download it first.")
        return

    model = joblib.load(MODEL_FILENAME)
    landmarker = _build_live_landmarker()

    colors = {
        "ok":      (0, 255, 0),
        "peace":   (255, 180, 0),
        "stop":    (0, 0, 255),
        "like":    (0, 200, 255),
        "dislike": (255, 0, 180),
        "unknown": (128, 128, 128),
    }

    print("Opening live camera feed — press Q to quit.")
    cap = cv2.VideoCapture(0)
    timestamp_ms = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        features, rect, lm_list = _extract_live_landmarks(landmarker, frame_rgb, timestamp_ms)
        timestamp_ms += 33  # ~30 fps

        if features is not None:
            proba = model.predict_proba([features])[0]
            top_idx = int(np.argmax(proba))
            confidence = float(proba[top_idx])

            if confidence >= 0.5:
                gesture = CATEGORIES[top_idx]
                label = f"{gesture.upper()}  {confidence*100:.0f}%"
                color = colors[gesture]
            else:
                gesture = "unknown"
                label = f"UNKNOWN  {confidence*100:.0f}%"
                color = colors["unknown"]

            # Draw skeleton
            _draw_hand_connections(frame, lm_list)

            # Draw bounding box and label
            x1, y1, x2, y2 = rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1 - 35), (x2, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Show per-class probability bar
            bar_x = 10
            bar_y = frame.shape[0] - len(CATEGORIES) * 30 - 10
            for i, cat in enumerate(CATEGORIES):
                pct = int(proba[i] * 150)
                cv2.rectangle(frame, (bar_x, bar_y + i*28),
                              (bar_x + pct, bar_y + i*28 + 18), colors[cat], -1)
                cv2.putText(frame, f"{cat}: {proba[i]*100:.0f}%",
                            (bar_x + 158, bar_y + i*28 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "No hand detected", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, colors["unknown"], 2)

        cv2.imshow("Hand Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    print("=== Running Live Inference System ===")
    run_inference()
