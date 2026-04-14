from typing import List, Optional

from pydantic import BaseModel, Field

class DocumentRecord(BaseModel):
    doc_id: str
    title: str
    source_url: str
    category: str
    section_path: str
    parent_section: Optional[str] = None
    version: Optional[str] = None
    last_updated: Optional[str] = None
    raw_text: str

class ChunkRecord(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    chunk_text: str
    section_title: str
    section_path: str
    token_count: int
    embedding: Optional[List[float]] = None
    
class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source_url: str
    section_title: str
    section_path: str
    content: str
    retrieval_score: float
    retrieval_method: str
    rank: int

class Citation(BaseModel):
    citation_id: int
    chunk_id: str
    doc_id: str
    title: str
    source_url: str
    section_title: str
    snippet: str

class AnswerResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    query_type: Optional[str] = None
    retrieval_strategy: str
    confidence: float
    abstained: bool = False
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)

class SearchResponse(BaseModel):
    query: str
    retrieval_method: str
    top_k: int
    results: List[RetrievedChunk] = Field(default_factory=list)
