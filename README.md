# Evaluation-Driven RAG System

A production-oriented Retrieval-Augmented Generation (RAG) system for technical documentation, designed around retrieval experimentation, grounded answers, citation support, and evaluation-driven optimization.

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

- Markdown document parsing
- Text cleaning
- Heading-aware chunking
- Chunk metadata preservation, including section titles and section paths

### Embedding

- Local embedding pipeline using `sentence-transformers`
- Configurable embedding model and embedding dimensions
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
- persisted experiment outputs in JSON and CSV

Current metrics include:

- Hit@k
- Top-1 retrieval accuracy
- Abstention accuracy
- Citation presence accuracy
- Answer-level pass rate

### Failure Analysis

The project tracks observed failures and fixes in `docs/failure_analysis.md`.

One documented example is a hybrid retrieval false positive on an out-of-domain query, caused by forced ranking in dense and BM25 retrieval. The system was improved by adding evidence sufficiency checks for hybrid answers.

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
│   └── failure_analysis.md
├── scripts/
│   ├── create_tables.py
│   ├── run_ingestion_demo.py
│   ├── run_embedding.py
│   ├── run_retrieval_eval.py
│   └── run_answer_eval.py
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

### 5. Run demo ingestion

```bash
PYTHONPATH=. python scripts/run_ingestion_demo.py
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
    "query": "When should I use a DaemonSet?",
    "method": "hybrid",
    "top_k": 3
  }'
```

## Example Answer Request

```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "When should I use a DaemonSet?",
    "method": "hybrid",
    "top_k": 3
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

Evaluation outputs are saved to `data/experiments/`.

Generated experiment files are ignored by Git by default.

## Current Development Status

Completed:

- PostgreSQL + pgvector setup
- document and chunk schema
- ingestion demo
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
- failure analysis documentation

Next planned improvements:

- expand the evaluation dataset
- add real Kubernetes documentation ingestion
- improve BM25 tokenization and lexical evidence filtering
- add query classification
- add experiment comparison reports
- integrate LLM-based grounded answer generation
- add RAGAS-based evaluation