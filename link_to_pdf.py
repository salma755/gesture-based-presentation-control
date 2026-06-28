import time
import fitz
import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from settings import MODEL_FILENAME, CATEGORIES, LANDMARK_MODEL_PATH

# ضع مسار ملف الـ PDF هنا
PDF_PATH = "./chapter1.pdf"


# ── الخطوة 2: قراءة الـ PDF وتحويل صفحاته لصور ──────────────────────────────

def load_pdf_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    mat = fitz.Matrix(1.5, 1.5)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pages.append(img)
    return pages


# ── الخطوة 1: بناء الـ landmarker (مستعار من main.py) ───────────────────────

def _build_landmarker():
    base_opts = mp_python.BaseOptions(model_asset_path=LANDMARK_MODEL_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(opts)


def _extract_landmarks(landmarker, frame_rgb, timestamp_ms):
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect_for_video(mp_img, timestamp_ms)
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    origin = coords[0].copy()
    coords -= origin
    scale = np.linalg.norm(coords[9])
    if scale > 0:
        coords /= scale
    return coords.flatten()


# ── الخطوة 3 + 4: الحلقة الرئيسية ───────────────────────────────────────────

def run(pdf_path):
    pages = load_pdf_pages(pdf_path)
    total_pages = len(pages)
    print(f"PDF محمّل: {total_pages} صفحة")

    model = joblib.load(MODEL_FILENAME)
    landmarker = _build_landmarker()

    current_page = 0
    last_action_time = 0
    COOLDOWN = 1.5
    timestamp_ms = 0

    cap = cv2.VideoCapture(0)
    print("الكاميرا تعمل — حرّك يدك للتحكم بالـ PDF، واضغط Q للخروج.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features = _extract_landmarks(landmarker, frame_rgb, timestamp_ms)
        timestamp_ms += 33

        gesture = "unknown"
        if features is not None:
            proba = model.predict_proba([features])[0]
            top_idx = int(np.argmax(proba))
            if float(proba[top_idx]) >= 0.5:
                gesture = CATEGORIES[top_idx]

        # ── الخطوة 3: ربط الإيماءة بالأمر ───────────────────────────────────
        now = time.time()
        if gesture != "unknown" and now - last_action_time > COOLDOWN:
            if gesture == "like":
                current_page = min(current_page + 1, total_pages - 1)
                last_action_time = now
            elif gesture == "dislike":
                current_page = max(current_page - 1, 0)
                last_action_time = now
            elif gesture == "ok":
                current_page = 0
                last_action_time = now
            elif gesture == "peace":
                current_page = total_pages - 1
                last_action_time = now
            elif gesture == "stop":
                break

        # ── الخطوة 4: عرض الـ PDF + الكاميرا ────────────────────────────────
        pdf_display = pages[current_page].copy()
        label = f"Pg {current_page + 1}/{total_pages}  |  {gesture.upper()}"
        cv2.putText(pdf_display, label, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)
        cv2.imshow("PDF Viewer", pdf_display)

        cam_label = f"{gesture.upper()}"
        cv2.putText(frame, cam_label, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 0), 2)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    run(PDF_PATH)
