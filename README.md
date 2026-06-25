# gesture-based-presentation-control
Hand gesture recognition system using MediaPipe and machine learning to control presentation slides in real time.
# Hand Gesture Recognition & Slide Control System

A final project developed for the Pattern Recognition course. The system uses hand gesture recognition to classify predefined gestures and control presentation slides in real time.

The project integrates computer vision, feature extraction, and machine learning to build an interactive human-computer interface.


<img width="667" height="308" alt="Screenshot 2026-06-25 035721" src="https://github.com/user-attachments/assets/f76b1935-1271-40fe-be7a-7e7f2465247b" />

## Project Team

Developed collaboratively with:
- Dina Al-Shamayleh
- Sofyan Al-Tarawneh

## Project Idea

The system recognizes predefined hand gestures and maps them to actions used in presentation control.

Detected gestures include:

- data (neutral state)
- test
- like
- dislike
- ok
- peace
- stop
- train
<img width="285" height="336" alt="photo_2026-06-25_04-06-10" src="https://github.com/user-attachments/assets/24446b49-4de6-4a4c-823f-974e3558da64" />

## System Features

- Real-time hand tracking using MediaPipe
- Feature extraction from hand landmarks
- Gesture classification model trained on Kaggle dataset
- Mapping gestures to slide control actions
- Live interaction with presentation system

## Dataset

The dataset was collected and adapted from Kaggle and included multiple gesture classes such as:

- like / dislike
- ok
- peace
- stop
- train
- test
- data (baseline/neutral class)

## Methodology

1. Hand detection using MediaPipe
2. Extraction of hand landmarks (feature vectors)
3. Data preprocessing and normalization
4. Training a classification model on labeled gestures
5. Real-time prediction and action mapping

## Applications

- Presentation control using gestures
- Human-computer interaction systems
- Assistive technologies
- Smart interfaces without physical input devices

## Challenges

- Variability in hand positions and orientations
- Similarity between certain gesture classes
- Real-time prediction constraints
- Ensuring stable detection using MediaPipe landmarks

Despite these challenges, the system achieved functional real-time gesture recognition suitable for interactive slide control.

## Technologies

- Python
- MediaPipe
- OpenCV
- Scikit-learn / Machine Learning
- NumPy

## Learning Outcomes

- Real-time computer vision systems
- Feature extraction from spatial landmarks
- Gesture classification pipelines
- Human-computer interaction design
- Integration of ML with live applications
