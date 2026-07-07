from pydantic import BaseModel

from src.core.models import AnswerResponse
from src.generation.citation_builder import extract_citation_ids


class CitationEvalResult(BaseModel):
    referenced_citation_ids: list[int]
    returned_citation_ids: list[int]
    invalid_citation_ids: list[int]
    missing_returned_citation_ids: list[int]
    unreferenced_returned_citation_ids: list[int]
    citation_ids_valid: bool
    citation_alignment_correct: bool
    citation_utilization: float


def evaluate_citations(response: AnswerResponse) -> CitationEvalResult:
    referenced_ids = sorted(extract_citation_ids(response.answer))
    returned_ids = sorted(
        citation.citation_id
        for citation in response.citations
    )

    referenced_set = set(referenced_ids)
    returned_set = set(returned_ids)

    invalid_ids = sorted(
        citation_id
        for citation_id in referenced_set
        if citation_id <= 0
    )

    missing_returned_ids = sorted(referenced_set - returned_set)
    unreferenced_returned_ids = sorted(returned_set - referenced_set)

    citation_ids_valid = not invalid_ids and not missing_returned_ids
    citation_alignment_correct = referenced_set == returned_set

    citation_utilization = (
        len(referenced_set & returned_set) / len(returned_set)
        if returned_set
        else 0.0
    )

    return CitationEvalResult(
        referenced_citation_ids=referenced_ids,
        returned_citation_ids=returned_ids,
        invalid_citation_ids=invalid_ids,
        missing_returned_citation_ids=missing_returned_ids,
        unreferenced_returned_citation_ids=unreferenced_returned_ids,
        citation_ids_valid=citation_ids_valid,
        citation_alignment_correct=citation_alignment_correct,
        citation_utilization=citation_utilization,
    )