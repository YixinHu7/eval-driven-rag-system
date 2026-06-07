# Evaluation-Driven RAG System

A production-oriented Retrieval-Augmented Generation (RAG) system for technical documentation, designed around retrieval experimentation, grounded answers, citation support, abstention, and evaluation-driven optimization.

This project is not intended to be a simple chatbot demo. It is built as a modular RAG platform where retrieval strategies, answer behavior, and failure modes can be evaluated and improved systematically.

## Project Goals

The goal of this project is to build a high-quality RAG pipeline over complex technical documentation with:

- structured document ingestion
- PostgreSQL + pgvector storage
- dense vector retrieval
- BM25 lexical retrieval
- hybrid retrieval with Reciprocal Rank Fusion
- citation-grounded answer generation
- confidence-based abstention
- retrieval-level evaluation
- answer-level evaluation
- failure analysis and iterative system improvement

## Current Features

### Storage and Data Modeling

- Dockerized PostgreSQL database
- pgvector extension for vector similarity search
- SQLAlchemy ORM models for documents and chunks
- Repository layer for document and chunk operations

### Ingestion

- Real Kubernetes documentation ingestion
- Markdown and web document parsing
- Text cleaning
- Heading-aware chunking
- Chunk metadata preservation, including section titles and section paths

### Embedding

- Local embedding pipeline using `sentence-transformers`
- Configurable embedding model and embedding dimensions
- Metadata-enriched chunk representation
- Embeddings stored in PostgreSQL using pgvector

### Retrieval

The system currently supports three retrieval strategies:

- Dense retrieval using pgvector cosine distance
- BM25 lexical retrieval
- Hybrid retrieval using Reciprocal Rank Fusion

The retriever factory allows retrieval methods to be selected through a unified interface.

### API

The system exposes FastAPI endpoints:

- `GET /health`
- `POST /search`
- `POST /answer`

The `/search` endpoint returns structured retrieval results.

The `/answer` endpoint returns grounded answers with citations, confidence, abstention status, and retrieved chunks.

### Evaluation

The project includes custom evaluation runners for:

- retrieval-level evaluation
- answer-level evaluation
- query-type sliced metrics
- document-aware retrieval evaluation
- multi-accepted-section evaluation
- persisted experiment outputs in JSON and CSV
- retrieval failure inspection reports

Current metrics include:

- Hit@k
- Top-1 retrieval accuracy
- Abstention accuracy
- Citation presence accuracy
- Answer-level pass rate

## Current Best Results

The current benchmark uses a 24-question Kubernetes documentation evaluation set across four query types:

- conceptual
- procedural
- constraint
- out_of_domain

The best current retrieval strategy is hybrid retrieval with dense vector search and BM25 lexical search.

### Retrieval-Level Metrics

| Method | Hit@k | Top-1 Accuracy |
| ------ | ----: | -------------: |
| Dense  | 0.778 |          0.389 |
| BM25   | 0.556 |          0.278 |
| Hybrid | 0.722 |          0.500 |

### Key Findings

- Dense retrieval has the strongest overall Hit@k.
- BM25 token normalization significantly reduced wrong-document failures.
- Hybrid retrieval achieved the best Top-1 accuracy after BM25 normalization.
- Simple heuristic reranking was tested but did not improve performance.
- Multi-accepted-section evaluation made the benchmark more realistic by allowing multiple valid evidence sections.

## Experiment Tracking

Experiment notes are tracked in:

```text
docs/experiment_log.md
```

The experiment log records the hypothesis, implementation change, metrics, and conclusion for each retrieval optimization.

Current documented experiments include:

- metadata-enriched chunk representation
- heuristic reranking
- multi-accepted-section evaluation
- field-weighted BM25
- BM25 token normalization

## Failure Analysis

Failure analysis is tracked in:

```text
docs/failure_analysis.md
```

The project uses retrieval failure inspection reports to categorize errors such as:

- wrong document retrieval
- right document but wrong section
- expected section found in top-k but not ranked first
- out-of-domain false positives

## Architecture

```text
User Query
   ↓
FastAPI Endpoint
   ↓
Retriever Factory
   ├── Dense Retriever
   ├── BM25 Retriever
   └── Hybrid RRF Retriever
   ↓
Retrieved Chunks
   ↓
Answer Generator
   ├── Citation Builder
   └── Abstention Logic
   ↓
AnswerResponse
   ├── answer
   ├── citations
   ├── confidence
   ├── abstained
   └── retrieved_chunks
```

## Repository Structure

```text
eval-driven-rag-system/
├── configs/
├── data/
│   ├── raw/
│   ├── eval/
│   └── experiments/
├── docs/
│   ├── evaluation_report.md
│   ├── experiment_log.md
│   └── failure_analysis.md
├── scripts/
│   ├── create_tables.py
│   ├── ingest_k8s_docs.py
│   ├── run_answer_eval.py
│   ├── run_embedding.py
│   ├── run_ingestion_demo.py
│   ├── run_retrieval_eval.py
│   └── inspect_retrieval_failures.py
├── src/
│   ├── api/
│   ├── core/
│   ├── embedding/
│   ├── evaluation/
│   ├── generation/
│   ├── ingestion/
│   ├── retrieval/
│   └── storage/
└── tests/
```

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Start PostgreSQL with pgvector

```bash
docker compose up -d
```

The database runs on port `5433` to avoid conflicts with local PostgreSQL installations.

### 3. Configure environment variables

Create a `.env` file:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5433/eval_driven_rag
```

### 4. Create database tables

```bash
export $(grep -v '^#' .env | xargs)
PYTHONPATH=. python scripts/create_tables.py
```

### 5. Ingest Kubernetes documentation

```bash
PYTHONPATH=. python scripts/ingest_k8s_docs.py
```

### 6. Generate embeddings

```bash
PYTHONPATH=. python scripts/run_embedding.py
```

### 7. Start the API

```bash
PYTHONPATH=. uvicorn src.api.main:app --reload --port 8000
```

Open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

## Example Search Request

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do Kubernetes readiness probes work?",
    "method": "hybrid",
    "top_k": 5
  }'
```

## Example Answer Request

```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do Kubernetes readiness probes work?",
    "method": "hybrid",
    "top_k": 5
  }'
```

## Evaluation

Run retrieval-level evaluation:

```bash
PYTHONPATH=. python scripts/run_retrieval_eval.py
```

Run answer-level evaluation:

```bash
PYTHONPATH=. python scripts/run_answer_eval.py
```

Inspect retrieval failures:

```bash
PYTHONPATH=. python scripts/inspect_retrieval_failures.py
```

Evaluation outputs are saved to:

```text
data/experiments/
```

Generated experiment files are ignored by Git by default.

## Current Development Status

Completed:

- PostgreSQL + pgvector setup
- document and chunk schema
- real Kubernetes documentation ingestion
- local embedding pipeline
- dense retrieval
- BM25 retrieval
- hybrid retrieval
- retriever factory
- `/search` API
- `/answer` API
- abstention logic
- retrieval evaluation runner
- answer evaluation runner
- document-aware retrieval evaluation
- multi-accepted-section evaluation
- retrieval failure inspection reports
- failure analysis documentation
- experiment log documentation
- BM25 token normalization

Next planned improvements:

- improve section-level ranking for right-document-wrong-section failures
- add stronger BM25 tokenization and stopword handling
- add query classification for out-of-domain detection
- add experiment comparison reports
- integrate LLM-based grounded answer generation
- add RAGAS-based evaluation
- add tests for core retrieval and evaluation modules