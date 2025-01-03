from fastapi import FastAPI
from app.routers import dataset, model, metrics, prediction

def create_app() -> FastAPI:
    app = FastAPI(
        title="API de ML - Exemplo",
        description="Projeto FastAPI para ML",
        version="1.0.0"
    )

    app.include_router(dataset.router)
    app.include_router(model.router)
    app.include_router(metrics.router)
    app.include_router(prediction.router)

    return app

app = create_app()
