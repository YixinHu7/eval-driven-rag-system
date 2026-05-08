import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class EvalQuestion(BaseModel):
    question_id: str
    query: str
    query_type: str
    expected_section: Optional[str] = None
    expected_doc_id: Optional[str] = None
    supported: bool
    reference_notes: str


def load_eval_questions(path: str | Path) -> list[EvalQuestion]:
    eval_path = Path(path)

    with eval_path.open("r", encoding="utf-8") as f:
        raw_items = json.load(f)

    return [EvalQuestion(**item) for item in raw_items]