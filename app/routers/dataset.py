from fastapi import APIRouter, File, UploadFile, HTTPException
import os

from app.core.config import DATA_DIR
from app.core.utils import save_uploaded_file

router = APIRouter(prefix="/datasets", tags=["Datasets"])

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Endpoint to upload dataset."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    file_path = os.path.join(DATA_DIR, file.filename)
    save_uploaded_file(file, file_path)

    return {"message": "File uploaded successfully", "file_path": file_path}
