from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.routes import answer as answer_route
from src.api.routes import search as search_route
from src.core.models import (
    AnswerResponse,
    Citation,
    RetrievedChunk,
)


client = TestClient(create_app())


class FakeSession:
    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> bool:
        return False


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        session: Any,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        self.calls.append(
            {
                "session": session,
                "query": query,
                "top_k": top_k,
            }
        )
        return list(self.chunks)


class FakeGenerator:
    def __init__(self, response: AnswerResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        retrieval_strategy: str,
        query_type: str,
    ) -> AnswerResponse:
        self.calls.append(
            {
                "query": query,
                "chunks": chunks,
                "retrieval_strategy": retrieval_strategy,
                "query_type": query_type,
            }
        )
        return self.response


def make_chunk(
    chunk_id: str = "chunk_1",
    section_title: str = "Service",
    rank: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="k8s_service",
        title="Service",
        source_url="https://example.com/service",
        section_title=section_title,
        section_path=f"Service > {section_title}",
        content=f"Content for {section_title}.",
        retrieval_score=0.9,
        retrieval_method="hybrid",
        rank=rank,
    )


def make_supported_response(
    chunks: list[RetrievedChunk],
) -> AnswerResponse:
    first_chunk = chunks[0]

    return AnswerResponse(
        answer="A Service exposes an application. [1]",
        citations=[
            Citation(
                citation_id=1,
                chunk_id=first_chunk.chunk_id,
                doc_id=first_chunk.doc_id,
                title=first_chunk.title,
                source_url=first_chunk.source_url,
                section_title=first_chunk.section_title,
                snippet=first_chunk.content,
            )
        ],
        query_type="conceptual",
        retrieval_strategy="hybrid",
        confidence=0.9,
        abstained=False,
        retrieved_chunks=chunks,
    )


def make_abstention_response() -> AnswerResponse:
    return AnswerResponse(
        answer="I cannot answer this question from the available documentation.",
        citations=[],
        query_type="out_of_scope",
        retrieval_strategy="hybrid",
        confidence=0.0,
        abstained=True,
        retrieved_chunks=[],
    )


def patch_supported_query(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        answer_route,
        "classify_query",
        lambda query: SimpleNamespace(
            query_type="conceptual",
        ),
    )
    monkeypatch.setattr(
        answer_route,
        "should_short_circuit_answer",
        lambda classification: False,
    )


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_answer_rejects_empty_query() -> None:
    response = client.post(
        "/answer",
        json={
            "query": "",
            "method": "hybrid",
            "top_k": 5,
            "generator": "simple",
        },
    )

    assert response.status_code == 422


def test_answer_short_circuits_without_retrieval(
    monkeypatch: Any,
) -> None:
    abstention_response = make_abstention_response()

    monkeypatch.setattr(
        answer_route,
        "classify_query",
        lambda query: SimpleNamespace(
            query_type="out_of_scope",
        ),
    )
    monkeypatch.setattr(
        answer_route,
        "should_short_circuit_answer",
        lambda classification: True,
    )
    monkeypatch.setattr(
        answer_route,
        "build_routing_abstention_response",
        lambda classification, retrieval_strategy: abstention_response,
    )

    def fail_get_retriever(method: str) -> None:
        raise AssertionError(
            "Retriever should not be created for a short-circuited query."
        )

    monkeypatch.setattr(
        answer_route,
        "get_retriever",
        fail_get_retriever,
    )

    response = client.post(
        "/answer",
        json={
            "query": "Who won the World Cup?",
            "method": "hybrid",
            "top_k": 5,
            "generator": "simple",
        },
    )

    assert response.status_code == 200

    response_body = response.json()

    assert response_body["abstained"] is True
    assert response_body["citations"] == []
    assert response_body["retrieved_chunks"] == []


def test_answer_uses_initial_chunks_when_expansion_is_disabled(
    monkeypatch: Any,
) -> None:
    patch_supported_query(monkeypatch)

    seed_chunk = make_chunk()
    retriever = FakeRetriever([seed_chunk])
    generator = FakeGenerator(
        make_supported_response([seed_chunk])
    )

    monkeypatch.setattr(
        answer_route,
        "get_retriever",
        lambda method: retriever,
    )
    monkeypatch.setattr(
        answer_route,
        "SessionLocal",
        lambda: FakeSession(),
    )
    monkeypatch.setattr(
        answer_route,
        "SimpleAnswerGenerator",
        lambda: generator,
    )
    monkeypatch.setattr(
        answer_route.settings.generation,
        "enable_context_expansion",
        False,
    )

    def fail_expansion(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            "Context expansion should not run when disabled."
        )

    monkeypatch.setattr(
        answer_route,
        "expand_with_neighbor_chunks",
        fail_expansion,
    )

    response = client.post(
        "/answer",
        json={
            "query": "What is a Kubernetes Service?",
            "method": "hybrid",
            "top_k": 3,
            "generator": "simple",
        },
    )

    assert response.status_code == 200
    assert response.json()["abstained"] is False
    assert len(response.json()["citations"]) == 1

    assert len(retriever.calls) == 1
    assert retriever.calls[0]["query"] == (
        "What is a Kubernetes Service?"
    )
    assert retriever.calls[0]["top_k"] == 3

    assert len(generator.calls) == 1
    assert generator.calls[0]["chunks"] == [
        seed_chunk
    ]
    assert generator.calls[0]["retrieval_strategy"] == "hybrid"
    assert generator.calls[0]["query_type"] == "conceptual"


def test_answer_passes_expanded_chunks_to_generator(
    monkeypatch: Any,
) -> None:
    patch_supported_query(monkeypatch)

    seed_chunk = make_chunk(
        chunk_id="seed_chunk",
        section_title="Pod Identity",
        rank=1,
    )
    neighbor_chunk = make_chunk(
        chunk_id="neighbor_chunk",
        section_title="Ordinal Index",
        rank=2,
    )

    retriever = FakeRetriever([seed_chunk])
    generator = FakeGenerator(
        make_supported_response(
            [seed_chunk, neighbor_chunk]
        )
    )
    expansion_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        answer_route,
        "get_retriever",
        lambda method: retriever,
    )
    monkeypatch.setattr(
        answer_route,
        "SessionLocal",
        lambda: FakeSession(),
    )
    monkeypatch.setattr(
        answer_route,
        "SimpleAnswerGenerator",
        lambda: generator,
    )
    monkeypatch.setattr(
        answer_route.settings.generation,
        "enable_context_expansion",
        True,
    )

    def fake_expand_with_neighbor_chunks(
        session: Any,
        chunks: list[RetrievedChunk],
        window: int,
        max_chunks: int,
        max_neighbor_context_chars: int,
    ) -> list[RetrievedChunk]:
        expansion_calls.append(
            {
                "session": session,
                "chunks": chunks,
                "window": window,
                "max_chunks": max_chunks,
                "max_neighbor_context_chars": (
                    max_neighbor_context_chars
                ),
            }
        )
        return [
            seed_chunk,
            neighbor_chunk,
        ]

    monkeypatch.setattr(
        answer_route,
        "expand_with_neighbor_chunks",
        fake_expand_with_neighbor_chunks,
    )

    response = client.post(
        "/answer",
        json={
            "query": "How is StatefulSet Pod identity assigned?",
            "method": "hybrid",
            "top_k": 1,
            "generator": "simple",
        },
    )

    assert response.status_code == 200
    assert len(expansion_calls) == 1

    expansion_call = expansion_calls[0]

    assert expansion_call["chunks"] == [
        seed_chunk
    ]
    assert expansion_call["window"] == (
        answer_route.settings.generation.context_expansion_window
    )
    assert expansion_call["max_chunks"] == (
        answer_route.settings.generation.max_expanded_context_chunks
    )
    assert expansion_call["max_neighbor_context_chars"] == (
        answer_route.settings.generation.max_neighbor_context_chars
    )

    assert len(generator.calls) == 1
    assert generator.calls[0]["chunks"] == [
        seed_chunk,
        neighbor_chunk,
    ]


def test_search_endpoint_returns_retrieved_chunks(
    monkeypatch: Any,
) -> None:
    chunks = [
        make_chunk(
            chunk_id="chunk_1",
            rank=1,
        ),
        make_chunk(
            chunk_id="chunk_2",
            rank=2,
        ),
    ]
    retriever = FakeRetriever(chunks)

    monkeypatch.setattr(
        search_route,
        "get_retriever",
        lambda method: retriever,
    )
    monkeypatch.setattr(
        search_route,
        "SessionLocal",
        lambda: FakeSession(),
    )

    response = client.post(
        "/search",
        json={
            "query": "Kubernetes Service",
            "method": "hybrid",
            "top_k": 2,
        },
    )

    assert response.status_code == 200

    response_body = response.json()

    assert response_body["query"] == "Kubernetes Service"
    assert response_body["retrieval_method"] == "hybrid"
    assert response_body["top_k"] == 2
    assert len(response_body["results"]) == 2

    assert len(retriever.calls) == 1
    assert retriever.calls[0]["top_k"] == 2