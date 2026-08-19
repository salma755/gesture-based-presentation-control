# ✋ Hand Gesture Recognition

> A real-time **hand gesture recognition** system that detects five hand gestures from a live webcam feed using **MediaPipe Hand Landmarker** and a **Random Forest** classifier. The project also includes a gesture-controlled PDF viewer that allows users to navigate PDF documents using hand movements.

Built with **Python, OpenCV, MediaPipe, Scikit-learn, NumPy, Joblib, and PyMuPDF**.

---
<img width="667" height="308" alt="Screenshot 2026-06-25 035721" src="https://github.com/user-attachments/assets/3cf14c7b-27c0-4280-b593-638f8380c2f4" />

## 📌 Overview

This project implements a complete machine-learning pipeline for recognizing hand gestures in real time.

Instead of using raw image pixels, the system extracts **21 3D hand landmarks** using MediaPipe. Each landmark contains:

* X coordinate
* Y coordinate
* Z coordinate

This produces:

```text
21 landmarks × 3 coordinates = 63 features
```

The landmarks are normalized relative to the wrist and scaled using the distance between the wrist and the middle-finger MCP landmark. The resulting 63-dimensional feature vector is passed to a **Random Forest classifier**.

The trained model can then be used for real-time webcam recognition and as a human-computer interaction system for controlling PDF documents.

---

# ✨ Key Features

*  **Real-Time Recognition** — recognizes gestures directly from a webcam.
*  **MediaPipe Hand Landmarks** — uses 21 hand landmarks instead of raw pixels.
*  **Random Forest Classification** — uses a Scikit-learn Random Forest model with feature scaling.
*  **Confidence Estimation** — displays the predicted gesture and class probabilities.
*  **Hand Skeleton Visualization** — displays detected hand landmarks and connections.
*  **Bounding Box** — highlights the detected hand.
*  **Landmark Normalization** — makes features less dependent on hand position and scale.
*  **Gesture-Controlled PDF Viewer** — uses gestures to navigate PDF pages.
*  **Complete ML Pipeline** — includes data loading, training, evaluation, and inference.
*  **Model Persistence** — trained model is saved using Joblib.

---

# 🤚 Supported Gestures

| Gesture | Label     | PDF Action         |
| ------- | --------- | ------------------ |
| 👌      | `ok`      | Jump to first page |
| ✌️      | `peace`   | Jump to last page  |
| ✋       | `stop`    | Quit PDF viewer    |
| 👍      | `like`    | Next page          |
| 👎      | `dislike` | Previous page      |

The gesture classes are defined in `settings.py`:

```python
CATEGORIES = ["ok", "peace", "stop", "like", "dislike"]
```

---

#  How It Works

The complete system follows this pipeline:

```text
                    Webcam / Dataset
                           │
                           ▼
                 MediaPipe Hand Landmarker
                           │
                           ▼
                    21 Hand Landmarks
                           │
                           ▼
                    XYZ Coordinates
                           │
                           ▼
                  Landmark Normalization
                           │
                           ▼
                  63-Dimensional Vector
                           │
                           ▼
                     StandardScaler
                           │
                           ▼
                  Random Forest Classifier
                           │
                    ┌──────┴──────┐
                    │             │
                    ▼             ▼
               Prediction      Probability
                    │             │
                    └──────┬──────┘
                           │
                           ▼
                    Gesture Output
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       Live Visualization         PDF Controller
```

---

#  Landmark Extraction

MediaPipe detects **21 landmarks** for the hand.

Each landmark contains three coordinates:

```text
(x, y, z)
```

Therefore:

```text
21 × 3 = 63 numerical features
```

The feature vector is created in `data_loader.py`.

---

#  Feature Normalization

Raw landmark coordinates are affected by the hand's position and size within the camera frame.

To reduce this dependency, the project applies two normalization steps.

## 1. Wrist-Centered Coordinates

The wrist is used as the origin.

```python
origin = coords[0].copy()
coords -= origin
```

This makes the representation relative to the wrist instead of the image coordinates.

---

## 2. Scale Normalization

The distance between the wrist and landmark `9` is used as the reference hand scale:

```python
scale = np.linalg.norm(coords[9])
```

If the scale is greater than zero:

```python
coords /= scale
```

The final normalized coordinates are flattened into a 63-dimensional feature vector.

---

# 🌲 Machine Learning Model

The classifier is implemented using a Scikit-learn Pipeline:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ))
])
```

### Model Configuration

| Parameter            |          Value |
| -------------------- | -------------: |
| Algorithm            |  Random Forest |
| Number of estimators |            200 |
| Maximum depth        |             20 |
| Random state         |             42 |
| Parallel jobs        |           `-1` |
| Feature scaling      | StandardScaler |

The complete pipeline is saved using Joblib:

```text
hand_gesture_sklearn.joblib
```

---

#  Prediction & Confidence

During real-time inference, the model calculates class probabilities using:

```python
model.predict_proba()
```

The class with the highest probability is selected.

The current confidence threshold is:

```text
0.50
```

If the highest probability is below 50%, the system reports:

```text
UNKNOWN
```

This prevents low-confidence predictions from immediately being treated as valid gestures.

---

# 📈 Live Visualization

The webcam interface provides visual feedback including:

* Detected gesture
* Confidence percentage
* Hand bounding box
* Hand skeleton
* Individual class probabilities

Example:

```text
LIKE 92%

ok        2%
peace     1%
stop      1%
like     92%
dislike   4%
```

The hand skeleton is drawn using the detected MediaPipe landmarks.

---

#  Gesture-Controlled PDF Viewer

The project includes a second application that connects gesture recognition with PDF navigation.

The module:

```text
link_to_pdf.py
```

allows users to control PDF pages without using a keyboard or mouse.

---

##  PDF Controls

| Gesture      | Action        |
| ------------ | ------------- |
| 👍 `like`    | Next page     |
| 👎 `dislike` | Previous page |
| 👌 `ok`      | First page    |
| ✌️ `peace`   | Last page     |
| ✋ `stop`     | Exit          |

A cooldown mechanism is used to prevent the same gesture from repeatedly triggering an action every frame.

Current cooldown:

```python
COOLDOWN = 1.5
```

---

# 📑 PDF Processing

The PDF viewer uses **PyMuPDF (`fitz`)**.

The PDF is loaded and each page is rendered as an image before being displayed through OpenCV.

The process is:

```text
PDF File
   │
   ▼
PyMuPDF
   │
   ▼
PDF Pages
   │
   ▼
Rendered Images
   │
   ▼
OpenCV Display
```

The PDF path is configured in `link_to_pdf.py`.

Example:

```python
PDF_PATH = "./chapter1.pdf"
```

---

# 📂 Project Structure

```text
hand-gesture-recognition/
│
├── Data/
│   ├── train/
│   │   ├── ok/
│   │   ├── peace/
│   │   ├── stop/
│   │   ├── like/
│   │   └── dislike/
│   │
│   └── test/
│       ├── ok/
│       ├── peace/
│       ├── stop/
│       ├── like/
│       └── dislike/
│
├── main.py
│   └── Real-time webcam gesture recognition
│
├── link_to_pdf.py
│   └── Gesture-controlled PDF viewer
│
├── train.py
│   └── Train the Random Forest model
│
├── test.py
│   └── Evaluate the trained model
│
├── data_loader.py
│   └── Load images and extract landmarks
│
├── model.py
│   └── Random Forest model architecture
│
├── settings.py
│   └── Project configuration
│
├── hand_landmarker.task
│   └── MediaPipe Hand Landmarker model
│
├── hand_gesture_sklearn.joblib
│   └── Trained classification pipeline
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🛠️ Tech Stack

| Technology        | Purpose                             |
| ----------------- | ----------------------------------- |
| **Python 3**      | Core programming language           |
| **MediaPipe**     | Hand landmark detection             |
| **OpenCV**        | Webcam processing and visualization |
| **Scikit-learn**  | Machine learning                    |
| **Random Forest** | Gesture classification              |
| **NumPy**         | Numerical feature processing        |
| **Joblib**        | Model persistence                   |
| **PyMuPDF**       | PDF loading and rendering           |

---

# ⚙️ Requirements

* Python 3.9+
* Webcam
* MediaPipe
* OpenCV
* Scikit-learn
* NumPy
* Joblib
* PyMuPDF

---



# 📦 MediaPipe Model

The project requires:

```text
hand_landmarker.task
```

Place it in the root directory:

```text
hand-gesture-recognition/
│
├── hand_landmarker.task
├── main.py
├── train.py
└── ...
```

The model path is configured in:

```python
LANDMARK_MODEL_PATH = "hand_landmarker.task"
```

---

#  Dataset

The dataset is divided into training and testing sets.

Recommended structure:

```text
Data/
│
├── train/
│   ├── ok/
│   ├── peace/
│   ├── stop/
│   ├── like/
│   └── dislike/
│
└── test/
    ├── ok/
    ├── peace/
    ├── stop/
    ├── like/
    └── dislike/
```

Each category folder should contain images representing the corresponding gesture.

---

#  Training

To train the Random Forest classifier:

```bash
python train.py
```

The training process:

```text
Training Images
      │
      ▼
MediaPipe Hand Detection
      │
      ▼
Landmark Extraction
      │
      ▼
Normalization
      │
      ▼
63-D Feature Vectors
      │
      ▼
Random Forest Training
      │
      ▼
Joblib Model
```

The trained model is saved as:

```text
hand_gesture_sklearn.joblib
```

---

#  Evaluation

To evaluate the trained model:

```bash
python test.py
```

The evaluation script calculates:

* Accuracy
* Precision
* Recall
* F1-score

The exact performance depends on the dataset used for training and testing.

Example:

```text
=== Model Evaluation Report ===

Overall Accuracy: XX.XX%

Classification Metrics per Category:

              precision    recall    f1-score
ok              ...
peace           ...
stop            ...
like            ...
dislike         ...
```

---

#  Real-Time Inference

Run:

```bash
python main.py
```

The application opens the webcam and performs real-time gesture recognition.

The pipeline is:

```text
Webcam
  ↓
OpenCV
  ↓
MediaPipe
  ↓
21 Landmarks
  ↓
Normalization
  ↓
Random Forest
  ↓
Gesture + Confidence
```

Press:

```text
Q
```

to quit.

---

# 📄 PDF Controller

To start the gesture-controlled PDF viewer:

```bash
python link_to_pdf.py
```

Make sure the PDF path is correctly configured:

```python
PDF_PATH = "./chapter1.pdf"
```

The application displays the PDF and camera feed simultaneously.

---

# 🔄 Complete Workflow

```text
                  ┌───────────────┐
                  │    Dataset    │
                  └───────┬───────┘
                          │
                          ▼
                ┌──────────────────┐
                │ MediaPipe        │
                │ Hand Landmarker  │
                └────────┬─────────┘
                         │
                         ▼
                 21 Hand Landmarks
                         │
                         ▼
                 Feature Extraction
                         │
                         ▼
                Normalization
                         │
                         ▼
                 63-D Features
                         │
                         ▼
                StandardScaler
                         │
                         ▼
              Random Forest Model
                         │
                ┌────────┴────────┐
                │                 │
                ▼                 ▼
            Evaluation        Real-Time
                │                 │
                │                 ▼
                │             Webcam
                │                 │
                │                 ▼
                │          Gesture Detection
                │                 │
                │          ┌──────┴──────┐
                │          │             │
                │          ▼             ▼
                │       Display       PDF Control
                │
                ▼
          Accuracy / Precision
          Recall / F1-Score
```

---

# ⚠️ Limitations

The current implementation has several limitations:

* Supports one hand at a time.
* Recognition quality depends on the training dataset.
* Poor lighting can affect hand landmark detection.
* Heavy hand occlusion may cause detection failure.
* Extreme hand rotations may reduce recognition accuracy.
* Only five gesture classes are currently supported.
* The confidence threshold is fixed at 50%.
* The PDF viewer renders document pages as images, which may require additional memory for large PDFs.
* Real-time performance depends on camera quality and computer hardware.

---

# 🔮 Future Improvements

Possible improvements include:

### 🤚 Gesture Recognition

* Support multiple hands.
* Add more gesture classes.
* Increase dataset diversity.
* Add data augmentation.
* Compare Random Forest with SVM, KNN, and neural networks.
* Add temporal gesture recognition.

### 🎥 Real-Time Performance

* Improve detection stability.
* Add FPS monitoring.
* Add configurable confidence thresholds.
* Optimize processing for lower-end hardware.

### 📄 PDF Controller

* Add zoom gestures.
* Add annotation gestures.
* Add presentation mode.
* Add customizable gesture mappings.
* Add previous/next document controls.

### 🧠 Machine Learning

* Add cross-validation.
* Generate confusion matrices.
* Perform hyperparameter optimization.
* Compare multiple classification algorithms.
* Analyze per-class performance.

---

# 🎓 Academic Context

This project was developed as a **Pattern Recognition project** demonstrating the application of machine learning and computer vision to real-time human-computer interaction.

It combines:

* Feature Extraction
* Landmark Detection
* Feature Normalization
* Machine Learning Classification
* Model Evaluation
* Real-Time Computer Vision
* Human-Computer Interaction

---

# 📌 Project Status

**Status: Completed Academic Project / Prototype**

The project provides a complete workflow from dataset processing and model training to real-time gesture recognition and gesture-controlled PDF navigation , with Dina and Sofyan 

