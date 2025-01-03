from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.core.utils import load_metrics

router = APIRouter(prefix="/metrics", tags=["Metrics"])

@router.get("/")
async def get_metrics(file_name: str = Query(...)):
    """Endpoint to fetch the metrics of a trained model."""
    metrics_str = load_metrics(file_name)
    return JSONResponse(content=metrics_str)
