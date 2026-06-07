import re

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.models import RetrievedChunk
from src.storage.schema import ChunkORM, DocumentORM
from src.retrieval.text_builder import build_chunk_search_text

STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "for",
    "and",
    "or",
    "in",
    "on",
    "with",
    "by",
    "how",
    "do",
    "i",
    "should",
    "when",
    "what",
    "why",
    "does",
    "did",
    "can",
    "could",
    "would",
    "you",
    "your",
}


NORMALIZATION_MAP = {
    "services": "service",
    "secrets": "secret",
    "configmaps": "configmap",
    "deployments": "deployment",
    "statefulsets": "statefulset",
    "daemonsets": "daemonset",
    "pods": "pod",
    "nodes": "node",
    "containers": "container",
    "probes": "probe",
    "rules": "rule",
    "values": "value",
    "variables": "variable",
    "labels": "label",
    "rollouts": "rollout",
}


def normalize_token(token: str) -> str:
    if token in NORMALIZATION_MAP:
        return NORMALIZATION_MAP[token]

    # conservative plural normalization
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]

    return token


def simple_tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())

    tokens: list[str] = []
    for token in raw_tokens:
        if token in STOPWORDS:
            continue

        normalized = normalize_token(token)

        if normalized and normalized not in STOPWORDS:
            tokens.append(normalized)

    return tokens


class BM25Retriever:
    def __init__(self) -> None:
        self.default_top_k = settings.retrieval.top_k

    def retrieve(
        self,
        session: Session,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        k = top_k or self.default_top_k

        stmt = (
            select(ChunkORM, DocumentORM)
            .join(DocumentORM, ChunkORM.doc_id == DocumentORM.doc_id)
            .order_by(ChunkORM.doc_id.asc(), ChunkORM.chunk_index.asc())
        )

        rows = session.execute(stmt).all()

        if not rows:
            return []

        corpus_tokens: list[list[str]] = []
        joined_rows: list[tuple[ChunkORM, DocumentORM]] = []

        for chunk, document in rows:
            search_text = build_chunk_search_text(chunk)
            corpus_tokens.append(simple_tokenize(search_text))
            joined_rows.append((chunk, document))

        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = simple_tokenize(query)
        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        results: list[RetrievedChunk] = []

        for rank, idx in enumerate(ranked_indices, start=1):
            chunk, document = joined_rows[idx]
            score = float(scores[idx])

            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=document.title,
                    source_url=document.source_url,
                    section_title=chunk.section_title,
                    section_path=chunk.section_path,
                    content=chunk.chunk_text,
                    retrieval_score=score,
                    retrieval_method="bm25",
                    rank=rank,
                )
            )

        return results