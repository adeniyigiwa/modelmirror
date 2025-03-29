import os
import joblib
import importlib.util
import onnxruntime as ort
import tempfile
import streamlit as st
import pandas as pd


def load_any_model(file) -> object:
    """
    Attempts to load a model from various formats:
    - .pkl (joblib)
    - .py  (expects a get_model() function)
    - .onnx (returns ONNX session)
    """
    try:
        file_name = file.name.lower()

        if file_name.endswith(".pkl"):
            return joblib.load(file)

        elif file_name.endswith(".py"):
            # Save file to temp directory and import
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            spec = importlib.util.spec_from_file_location("user_model", tmp_path)
            user_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(user_module)

            if hasattr(user_module, "get_model"):
                return user_module.get_model()
            else:
                raise AttributeError("Python file must define a get_model() function.")

        elif file_name.endswith(".onnx"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            session = ort.InferenceSession(tmp_path)
            return session

        else:
            raise ValueError("Unsupported model format. Upload a .pkl, .py, or .onnx file.")

    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None


def load_any_dataset(file) -> pd.DataFrame:
    """
    Load dataset from .csv, .xlsx, or .parquet
    """
    try:
        file_name = file.name.lower()

        if file_name.endswith(".csv"):
            return pd.read_csv(file)
        elif file_name.endswith(".xlsx"):
            return pd.read_excel(file)
        elif file_name.endswith(".parquet"):
            return pd.read_parquet(file)
        else:
            raise ValueError("Unsupported dataset format. Use .csv, .xlsx, or .parquet")

    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        return pd.DataFrame()


# In your app.py, replace model loading like this:
# from utils.load_model import load_any_model, load_any_dataset
# model = load_any_model(uploaded_model)
# data = load_any_dataset(uploaded_data)