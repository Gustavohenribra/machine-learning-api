import os
import shutil
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from fastapi import HTTPException

from .config import DATA_DIR, MODEL_DIR, RESULTS_DIR

def save_uploaded_file(upload_file, destination_path: str):
    """Salva um arquivo enviado via UploadFile no caminho especificado."""
    with open(destination_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

def train_random_forest(
    dataset_path: str,
    target_column: str,
    n_estimators: int = 100,
    max_depth: int = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Recebe parâmetros de hiperparâmetros e split,
    treina RandomForest e salva o modelo e métricas em disco.
    """
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found.")

    try:
        data = pd.read_csv(dataset_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading dataset: {e}")

    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail="Target column not found in dataset.")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    base_name = os.path.splitext(os.path.basename(dataset_path))[0]
    model_path = os.path.join(MODEL_DIR, f"model_{base_name}.joblib")
    joblib.dump(model, model_path)

    metrics_path = os.path.join(RESULTS_DIR, f"metrics_{base_name}.json")
    pd.DataFrame(report).to_json(metrics_path)

    return {
        "model_path": model_path,
        "report": report,
        "accuracy": accuracy,
        "metrics_path": metrics_path
    }

def load_model(model_name: str):
    """
    Recebe o nome do modelo (ex.: "model_flowers.joblib") e tenta carregar.
    """
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found.")
    return joblib.load(model_path)

def load_metrics(file_name: str) -> str:
    """
    Recebe o nome do arquivo CSV (ex.: "flowers.csv"), extrai o base_name ("flowers")
    e procura por "metrics_flowers.json" em RESULTS_DIR.
    """
    base_name = os.path.splitext(file_name)[0]
    metrics_path = os.path.join(RESULTS_DIR, f"metrics_{base_name}.json")

    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics not found.")

    with open(metrics_path, "r") as f:
        metrics_str = f.read()

    return metrics_str
