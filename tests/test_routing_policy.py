from src.routing.policy import (
    build_routing_abstention_response,
    should_short_circuit_answer,
)
from src.routing.query_classifier import QueryClassification


def test_short_circuits_out_of_domain_query() -> None:
    classification = QueryClassification(
        query_type="out_of_domain",
        confidence=0.7,
        matched_terms=[],
    )

    assert should_short_circuit_answer(classification) is True


def test_does_not_short_circuit_in_domain_query() -> None:
    classification = QueryClassification(
        query_type="in_domain",
        confidence=0.8,
        matched_terms=["kubernetes", "pod"],
    )

    assert should_short_circuit_answer(classification) is False


def test_build_routing_abstention_response() -> None:
    classification = QueryClassification(
        query_type="out_of_domain",
        confidence=0.7,
        matched_terms=[],
    )

    response = build_routing_abstention_response(
        classification=classification,
        retrieval_strategy="hybrid",
    )

    assert response.abstained is True
    assert response.confidence == 0.0
    assert response.citations == []
    assert response.retrieved_chunks == []
    assert response.query_type == "out_of_domain"
    assert response.retrieval_strategy == "hybrid"