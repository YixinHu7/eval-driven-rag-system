from sentence_transformers import SentenceTransformer

from src.core.config import settings


class LocalEmbedder:
    def __init__(self) -> None:
        self.model = SentenceTransformer(
            settings.embedding.model_name,
            device=settings.embedding.device,
        )
        self.normalize_embeddings = settings.embedding.normalize_embeddings

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings.tolist()