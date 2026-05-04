from fastapi import FastAPI

from src.core.config import settings
from src.api.routes.search import router as search_router


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app.project_name,
        version="0.1.0",
        description="Evaluation-driven RAG system for technical documentation.",
    )

    app.include_router(search_router)

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()