import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class EvalQuestion(BaseModel):
    question_id: str
    query: str
    query_type: str
    expected_section: Optional[str] = None
    expected_doc_id: Optional[str] = None
    accepted_doc_sections: list[str] = Field(default_factory=list)
    supported: bool
    reference_notes: str
    
    def get_accepted_doc_sections(self) -> list[str]:
        if self.accepted_doc_sections:
            return self.accepted_doc_sections

        if self.expected_doc_id and self.expected_section:
            return [f"{self.expected_doc_id}::{self.expected_section}"]

        return []


def load_eval_questions(path: str | Path) -> list[EvalQuestion]:
    eval_path = Path(path)

    with eval_path.open("r", encoding="utf-8") as f:
        raw_items = json.load(f)

    return [EvalQuestion(**item) for item in raw_items]