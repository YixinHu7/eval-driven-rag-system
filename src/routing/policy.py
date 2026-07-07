from src.core.models import AnswerResponse
from src.routing.query_classifier import QueryClassification


def should_short_circuit_answer(
    classification: QueryClassification,
) -> bool:
    return classification.query_type == "out_of_domain"


def build_routing_abstention_response(
    classification: QueryClassification,
    retrieval_strategy: str,
) -> AnswerResponse:
    return AnswerResponse(
        answer=(
            "This question appears to be outside the supported Kubernetes "
            "documentation scope, so I cannot answer it with the available context."
        ),
        citations=[],
        query_type=classification.query_type,
        retrieval_strategy=retrieval_strategy,
        confidence=0.0,
        abstained=True,
        retrieved_chunks=[],
    )