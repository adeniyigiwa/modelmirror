import joblib
import os

def load_model(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("Model file not found.")
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")