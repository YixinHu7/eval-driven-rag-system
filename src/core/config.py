import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import yaml

load_dotenv()

class AppConfig(BaseModel):
    project_name: str = "eval-driven-rag-system"
    environment: str = "development"
    log_level: str = "INFO"

class DataConfig(BaseModel):
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    chunks_dir: Path = Path("data/chunks")
    eval_dir: Path = Path("data/eval")
    experiments_dir: Path = Path("data/experiments")

class PostgresConfig(BaseModel):
    database_url: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@localhost:5432/eval_driven_rag",
        )
    )
    echo_sql: bool = False

class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-small-en-v1.5"
    device: str = "cpu"
    normalize_embeddings: bool = True
    embedding_dimensions: int = 384

class ChunkingConfig(BaseModel):
    strategy: str = "heading_aware"
    chunk_size: int = 500
    chunk_overlap: int = 100

class RetrievalConfig(BaseModel):
    default_method: str = "dense"
    top_k: int = 5
    candidate_k: int = 20
    enable_reranker: bool = False
    similarity_metric: str = "cosine"

class RerankerConfig(BaseModel):
    model_name: str = "BAAI/bge-reranker-base"
    device: str = "cpu"
    candidate_k: int = 20
    top_k: int = 5

class GenerationConfig(BaseModel):
    max_context_chunks: int = 5
    max_tokens: int = 512
    temperature: float = 0.0
    
    enable_context_expansion: bool = True
    context_expansion_window: int = 1
    max_expanded_context_chunks: int = 8

class EvalConfig(BaseModel):
    eval_data_path: Path = Path("data/eval/questions.json")
    enable_ragas: bool = False

class LLMConfig(BaseModel):
    provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    model_name: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    max_tokens: int = 512
    temperature: float = 0.0

class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

def load_settings(config_path: Optional[str] = None) -> Settings:
    if config_path is None:
        return Settings()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    return Settings(**raw_config)


settings = Settings()