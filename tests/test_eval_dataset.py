from pathlib import Path

from src.evaluation.dataset import load_eval_questions


def test_load_eval_questions(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        """
        [
          {
            "question_id": "q001",
            "query": "What is a Service?",
            "query_type": "conceptual",
            "expected_section": "Service",
            "expected_doc_id": "k8s_service",
            "accepted_doc_sections": ["k8s_service::Service"],
            "supported": true,
            "reference_notes": "A Service exposes applications."
          }
        ]
        """,
        encoding="utf-8",
    )

    questions = load_eval_questions(path)

    assert len(questions) == 1
    assert questions[0].question_id == "q001"
    assert questions[0].get_accepted_doc_sections() == ["k8s_service::Service"]


def test_eval_question_falls_back_to_expected_doc_section(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        """
        [
          {
            "question_id": "q001",
            "query": "What is a Service?",
            "query_type": "conceptual",
            "expected_section": "Service",
            "expected_doc_id": "k8s_service",
            "supported": true,
            "reference_notes": "A Service exposes applications."
          }
        ]
        """,
        encoding="utf-8",
    )

    question = load_eval_questions(path)[0]

    assert question.get_accepted_doc_sections() == ["k8s_service::Service"]