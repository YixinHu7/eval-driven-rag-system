from src.core.config import settings
from src.evaluation.dataset import load_eval_questions


def main() -> None:
    questions = load_eval_questions(settings.evaluation.eval_data_path)

    print(f"Loaded {len(questions)} evaluation questions")

    for question in questions:
        print("-" * 80)
        print(f"ID: {question.question_id}")
        print(f"Type: {question.query_type}")
        print(f"Supported: {question.supported}")
        print(f"Query: {question.query}")
        print(f"Expected section: {question.expected_section}")


if __name__ == "__main__":
    main()