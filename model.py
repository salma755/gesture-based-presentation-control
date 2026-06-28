from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def create_model():
    """Random Forest with feature scaling — better for landmark coordinates."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)),
    ])

if __name__ == "__main__":
    print("Building default model architecture...")
    model = create_model()
    print(f"Success: Model structure created successfully: {model}")
