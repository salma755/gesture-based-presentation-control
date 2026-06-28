import os
import joblib
from sklearn.metrics import classification_report, accuracy_score
from data_loader import load_data
from settings import MODEL_FILENAME, CATEGORIES

def run_evaluation():
    if not os.path.exists(MODEL_FILENAME):
        print(f"Error: Saved model '{MODEL_FILENAME}' not found. Please run train.py first.")
        return

    print("Loading the saved model...")
    model = joblib.load(MODEL_FILENAME)

    print("Loading evaluation data (Test) via Data Loader...")
    X_test, y_test = load_data("test")
    
    if len(X_test) == 0:
        print("Error: No test images found in 'Data/test' directory.")
        return
        
    print(f"Test data shape: {X_test.shape}")

    print("\nCalculating predictions and evaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n=== Model Evaluation Report ===")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Metrics per Category (Precision & Recall):")
    print(classification_report(y_test, y_pred, target_names=CATEGORIES))

if __name__ == "__main__":
    print("=== Starting Evaluation Phase ===")
    run_evaluation()
