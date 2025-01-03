from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
import os

from app.core.config import DATA_DIR, MODEL_DIR
from app.core.utils import train_random_forest, load_model
from app.schemas.train import TrainRequest

router = APIRouter(prefix="/model", tags=["Model"])

@router.post("/train")
async def train_model(payload: TrainRequest):
    """
    Endpoint to train a model with a specific dataset.
    Allows customizing some hyperparameters and data split.
    """
    dataset_full_path = os.path.join(DATA_DIR, payload.file_name)

    result = train_random_forest(
        dataset_path=dataset_full_path,
        target_column=payload.target_column,
        n_estimators=payload.n_estimators,
        max_depth=payload.max_depth,
        test_size=payload.test_size,
        random_state=payload.random_state
    )

    return {
        "message": "Model trained successfully",
        "model_path": result["model_path"],
        "metrics": result["report"],
        "accuracy": result["accuracy"],
        "test_size": payload.test_size,
        "n_estimators": payload.n_estimators,
        "max_depth": payload.max_depth,
    }


@router.get("/download-file")
async def download_model(model_name: str, request: Request):
    """Endpoint to get a download link for a trained model."""
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found.")

    base_url = f"{request.url.scheme}://{request.client.host}:{request.url.port}"
    download_url = f"{base_url}/model/download-direct/{model_name}"
    return {"message": "Model ready for download", "download_url": download_url}

@router.get("/download-direct/{model_name}")
async def serve_model(model_name: str):
    """Endpoint to serve the model file for direct download."""
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found.")

    return FileResponse(model_path, media_type="application/octet-stream", filename=model_name)
