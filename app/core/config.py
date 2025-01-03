import os

DATA_DIR = os.environ.get("DATA_DIR", "datasets")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
