from fastapi import APIRouter, Query, HTTPException
import numpy as np
import logging

from app.core.utils import load_model
from app.schemas.inference import InferenceRequest, BatchInferenceRequest

router = APIRouter(prefix="/predict", tags=["Prediction"])
logger = logging.getLogger(__name__)

@router.post("/")
async def predict_single(model_name: str = Query(...), request: InferenceRequest = None):
    """
    Endpoint to predict on a trained model for a single sample.
    Exemplo de JSON de entrada:
    {
      "features": [5.1, 3.5, 1.4, 0.2]
    }
    """
    try:
        model = load_model(model_name)
        
        features = np.array(request.features).reshape(1, -1)

        if hasattr(model, "n_features_in_"):
            if features.shape[1] != model.n_features_in_:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected {model.n_features_in_} features, got {features.shape[1]}"
                )

        prediction = model.predict(features)
        
        return {
            "prediction": prediction.tolist()
        }
    except Exception as e:
        logger.exception("Error during single prediction")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/batch")
async def predict_batch(
    model_name: str = Query(...), 
    request: BatchInferenceRequest = None
):
    """
    Endpoint to predict on a trained model with multiple samples (batch).
    Exemplo de JSON de entrada:
    {
      "features_batch": [
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 3.4, 5.4, 2.3]
      ]
    }
    """
    try:
        model = load_model(model_name)
        features_array = np.array(request.features_batch)

        if len(features_array.shape) != 2:
            raise ValueError("Input data must be a 2D array (list of lists).")
        
        if hasattr(model, "n_features_in_"):
            if features_array.shape[1] != model.n_features_in_:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected {model.n_features_in_} features, got {features_array.shape[1]}"
                )

        predictions = model.predict(features_array)
        
        return {
            "predictions": predictions.tolist()
        }
    except Exception as e:
        logger.exception("Error during batch prediction")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/probabilities")
async def predict_probabilities(
    model_name: str = Query(...), 
    request: BatchInferenceRequest = None
):
    try:
        model = load_model(model_name)

        if not hasattr(model, "predict_proba"):
            raise HTTPException(
                status_code=400,
                detail="This model does not support probability predictions."
            )

        features_array = np.array(request.features_batch)
        if len(features_array.shape) != 2:
            raise ValueError("Input data must be a 2D array (list of lists).")

        if hasattr(model, "n_features_in_"):
            if features_array.shape[1] != model.n_features_in_:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected {model.n_features_in_} features, got {features_array.shape[1]}"
                )

        proba_array = model.predict_proba(features_array)
        probabilities = proba_array.tolist()

        classes_ = getattr(model, "classes_", None)
        if classes_ is not None:
            if hasattr(classes_, "tolist"):
                classes_ = classes_.tolist()
            else:
                classes_ = list(classes_)

        return {
            "probabilities": probabilities,
            "classes": classes_
        }

    except Exception as e:
        logger.exception("Error during probability prediction")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/top-n")
async def predict_top_n(
    model_name: str = Query(...),
    request: BatchInferenceRequest = None,
    n: int = Query(3)
):
    """
    Endpoint para retornar as top-N classes com suas probabilidades
    (para problemas de classificação multiclasse).
    Exemplo de JSON de entrada (batch):
    {
      "features_batch": [
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 3.4, 5.4, 2.3]
      ]
    }
    Query param:  n=3 (top 3 classes)
    """
    try:
        model = load_model(model_name)

        if not hasattr(model, "predict_proba"):
            raise HTTPException(
                status_code=400,
                detail="This model doesn't support predict_proba."
            )

        features_array = np.array(request.features_batch)
        if len(features_array.shape) != 2:
            raise ValueError("Input data must be a 2D array (list of lists).")

        if hasattr(model, "n_features_in_"):
            if features_array.shape[1] != model.n_features_in_:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected {model.n_features_in_} features, got {features_array.shape[1]}"
                )

        proba = model.predict_proba(features_array)
        classes = getattr(model, "classes_", None)
        if classes is None:
            raise HTTPException(
                status_code=400,
                detail="Could not retrieve 'classes_' from this model."
            )

        top_n_results = []
        for row in proba:
            sorted_indices = np.argsort(row)[::-1]
            top_n_indices = sorted_indices[:n]
            top_n_info = [
                {
                    "class": str(classes[idx]),
                    "probability": float(row[idx])
                }
                for idx in top_n_indices
            ]
            top_n_results.append(top_n_info)

        return {
            "top_n_predictions": top_n_results
        }
    except Exception as e:
        logger.exception("Error during top-n prediction")
        raise HTTPException(status_code=400, detail=str(e))
