# Evaluation-Driven RAG System

A modular Retrieval-Augmented Generation system for technical documentation, built around measurable retrieval quality, grounded answer generation, citation validation, safe abstention, and reproducible experimentation.

This project is designed as more than a chatbot demonstration. Each major component—retrieval, routing, context construction, answer generation, citation handling, and abstention—is independently configurable and evaluated against a structured benchmark.

## Project Goals

The system is designed to provide:

* structured technical-document ingestion
* PostgreSQL and pgvector storage
* dense, lexical, and hybrid retrieval
* optional cross-encoder reranking
* query classification and early abstention
* grounded LLM answer generation
* inline citations tied to retrieved chunks
* citation validity and alignment checks
* multi-section evidence evaluation
* configurable context expansion
* reproducible retrieval and answer experiments
* failure analysis driven by evaluation results

## System Overview

The evaluated answer pipeline is:

```text
User query
   ↓
Query classification
   ↓
Routing policy
   ├── Unsupported or out-of-domain → early abstention
   └── Supported → retrieval
                     ↓
              Retriever factory
              ├── Dense
              ├── BM25
              ├── Hybrid RRF
              └── Optional reranked variants
                     ↓
              Optional neighbor expansion
                     ↓
              Context selection
                     ↓
              Answer generator
              ├── Simple generator
              └── Grounded LLM generator
                     ↓
              Citation filtering and validation
                     ↓
              AnswerResponse
```

## Current Features

### Storage and Data Modeling

* Dockerized PostgreSQL database
* pgvector extension for vector similarity search
* SQLAlchemy ORM models for documents and chunks
* repository layer for document and chunk operations
* chunk metadata including document ID, section title, section path, and chunk index

### Ingestion

* Kubernetes documentation ingestion
* Markdown and web-document parsing
* text cleaning
* heading-aware chunking
* configurable chunk size and overlap
* preservation of document and section metadata

### Embeddings

* local embeddings using `sentence-transformers`
* configurable embedding model and dimensions
* metadata-enriched embedding text
* normalized embeddings
* vector storage in PostgreSQL through pgvector

### Retrieval

The project supports:

* dense retrieval using pgvector cosine distance
* BM25 lexical retrieval
* hybrid retrieval using Reciprocal Rank Fusion
* optional dense and hybrid cross-encoder reranking
* BM25 token normalization for technical vocabulary
* document-aware and section-aware evaluation

The API currently accepts the following retrieval methods:

```text
dense
bm25
hybrid
```

Reranked retrievers are available through the evaluation and retriever-factory workflows.

### Query Routing and Abstention

Before retrieval, the system classifies the query and applies a routing policy.

Queries that are unsupported or outside the documentation domain can be rejected before retrieval and generation. This reduces unnecessary model calls and limits unsupported answers.

### Grounded Answer Generation

The system provides two generators:

* `simple`
* `llm`

The LLM generator:

* uses retrieved chunks as its answer context
* requires inline citation markers such as `[1]`
* filters returned citations to citations used by the answer
* fails safely when usable citations are absent
* returns structured confidence and abstention information

### Citation Evaluation

Citation evaluation includes:

* referenced citation IDs
* returned citation IDs
* invalid citation detection
* missing citation detection
* unreferenced returned citations
* citation ID validity
* citation alignment
* citation utilization
* answered-only citation utilization

### Context Selection

Retrieved context is filtered before generation to remove low-value documentation boilerplate, such as generic feedback or navigation sections.

Potentially important warning and caution sections are retained.

### Section-Neighbor Context Expansion

The system can optionally add neighboring chunks from the same document.

Expansion is controlled by:

* neighbor window
* maximum final chunk count
* maximum added neighbor-context characters

Initial retrieved chunks are preserved before neighbors are added.

Context expansion is disabled by default because the final experiments showed that unconditional expansion increased prompt size without improving answer quality under the default `top_k=5` configuration.

It remains available for narrow-retrieval and neighbor-dependent workflows.

## API

The FastAPI application exposes:

```text
GET  /health
POST /search
POST /answer
```

### Search Response

The `/search` endpoint returns:

* query
* retrieval method
* requested top-k
* ranked retrieved chunks

### Answer Response

The `/answer` endpoint returns:

* generated answer
* citations
* query type
* retrieval strategy
* confidence
* abstention status
* retrieved chunks

## Evaluation Framework

The project includes evaluation for:

* retrieval performance
* answer behavior
* query classification
* citation validity and alignment
* required multi-section evidence
* context-expansion quality and overhead
* query-type metric slices
* retrieval failure categorization

### Retrieval Metrics

* Hit@k
* Top-1 accuracy
* supported-question counts
* query-type slices
* wrong-document failures
* right-document-wrong-section failures

### Answer Metrics

* abstention accuracy
* citation presence accuracy
* answer-level pass rate
* citation ID validity
* citation alignment
* citation utilization
* answered-only citation utilization
* required evidence accuracy
* required evidence coverage

### Reproducible Experiment Matrix

The final matrix runner evaluates:

* Hybrid with default top-k and expansion enabled
* Hybrid with default top-k and expansion disabled
* targeted top-k 1 neighbor-dependent cases with expansion enabled
* targeted top-k 1 neighbor-dependent cases with expansion disabled
* Hybrid reranked with default top-k and expansion enabled

Each scenario writes its own JSON, CSV, and console output. The matrix runner also produces combined JSON and CSV summaries.

## Evaluation Dataset

The final answer benchmark contains:

| Dataset property                 | Count |
| -------------------------------- | ----: |
| Total questions                  |    26 |
| Supported questions              |    20 |
| Unsupported questions            |     6 |
| Required multi-section questions |     2 |

The benchmark covers:

* conceptual questions
* procedural questions
* constraint questions
* out-of-domain questions

The two targeted multi-section cases require evidence from adjacent documentation sections.

## Retrieval Results

| Method          | Hit@k | Top-1 Accuracy |
| --------------- | ----: | -------------: |
| Dense           | 0.778 |          0.389 |
| BM25            | 0.556 |          0.278 |
| Hybrid          | 0.722 |          0.500 |
| Hybrid reranked | 0.889 |          0.444 |

Hybrid retrieval remains the default strategy because it provides the strongest Top-1 result and a practical balance between lexical document routing and semantic retrieval.

Cross-encoder reranking improved Hit@k but did not improve Top-1 or final answer-level performance. It remains an optional experimental retriever.

## Final Answer Results

The default Hybrid retriever with the LLM generator achieved:

| Metric                                | Result |
| ------------------------------------- | -----: |
| Abstention accuracy                   |  1.000 |
| Citation presence accuracy            |  1.000 |
| Pass rate                             |  1.000 |
| Citation ID validity rate             |  1.000 |
| Citation alignment rate               |  1.000 |
| Average citation utilization          |  0.769 |
| Average answered citation utilization |  1.000 |
| Required evidence accuracy            |  1.000 |
| Average required evidence coverage    |  1.000 |

Overall citation utilization is `0.769` because 20 of the 26 questions were supported and returned citations. The six unsupported questions correctly abstained without citations.

Citation utilization among answered questions was `1.000`.

## Context Expansion Findings

### Default Top-k 5

With expansion enabled:

| Metric                     |  Result |
| -------------------------- | ------: |
| Pass rate                  |   1.000 |
| Required evidence accuracy |   1.000 |
| Average initial chunks     |   5.000 |
| Average final chunks       |   8.000 |
| Average added neighbors    |   3.000 |
| Average added characters   | 2191.95 |

With expansion disabled, answer and evidence metrics remained identical while no additional context was added.

### Targeted Top-k 1 Ablation

| Metric                     | Expansion Enabled | Expansion Disabled |
| -------------------------- | ----------------: | -----------------: |
| Pass rate                  |             1.000 |              0.000 |
| Required evidence accuracy |             1.000 |              0.000 |
| Required evidence coverage |             1.000 |              0.500 |
| Average added neighbors    |             2.000 |              0.000 |
| Average added characters   |           1543.50 |               0.00 |

This demonstrates that neighbor expansion is valuable when initial retrieval is narrow and the answer requires evidence from an adjacent chunk.

It is not used unconditionally under the default configuration because `top_k=5` already provides sufficient evidence on the current benchmark.

## Key Findings

* Metadata-enriched chunk representations improved dense retrieval.
* BM25 token normalization substantially reduced wrong-document failures.
* Hybrid retrieval achieved the strongest Top-1 accuracy.
* Lightweight heuristic reranking did not improve retrieval.
* Cross-encoder reranking improved Hit@k but not final answer performance.
* Query classification supports early abstention before retrieval.
* Grounded LLM generation achieved perfect citation alignment on the benchmark.
* Multi-section evidence metrics detect incomplete answers that ordinary citation-presence metrics miss.
* Neighbor expansion improves evidence completeness under constrained retrieval.
* Unconditional expansion increases prompt size without measurable benefit at the default top-k.

## Repository Structure

```text
eval-driven-rag-system/
├── data/
│   ├── eval/
│   ├── raw/
│   ├── processed/
│   ├── chunks/
│   └── experiments/
├── docs/
│   ├── evaluation_report.md
│   ├── experiment_log.md
│   ├── experiment_summary.md
│   └── failure_analysis.md
├── scripts/
│   ├── compare_retrievers.py
│   ├── create_tables.py
│   ├── ingest_k8s_docs.py
│   ├── inspect_neighbor_chunks.py
│   ├── inspect_retrieval_failures.py
│   ├── run_answer_eval.py
│   ├── run_embedding.py
│   ├── run_final_eval_matrix.py
│   ├── run_ingestion_demo.py
│   ├── run_query_classification_eval.py
│   └── run_retrieval_eval.py
├── src/
│   ├── api/
│   ├── core/
│   ├── embedding/
│   ├── evaluation/
│   ├── generation/
│   ├── ingestion/
│   ├── retrieval/
│   ├── routing/
│   └── storage/
├── tests/
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

Generated caches, local databases, ingested data, and experiment outputs are excluded through `.gitignore`.

## Setup

### 1. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Start PostgreSQL

```bash
docker compose up -d
```

The Dockerized PostgreSQL service is exposed on port `5433`.

Verify that it is available:

```bash
pg_isready -h localhost -p 5433
```

### 3. Configure Environment Variables

Create a `.env` file:

```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5433/eval_driven_rag

OPENAI_API_KEY=your_api_key
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```

`OPENAI_API_KEY` is required only for workflows that use the LLM generator.

### 4. Create Database Tables

```bash
PYTHONPATH=. python scripts/create_tables.py
```

### 5. Ingest Kubernetes Documentation

```bash
PYTHONPATH=. python scripts/ingest_k8s_docs.py
```

### 6. Generate Embeddings

```bash
PYTHONPATH=. python scripts/run_embedding.py
```

### 7. Start the API

```bash
PYTHONPATH=. uvicorn src.api.main:app \
  --reload \
  --port 8000
```

Interactive API documentation is available at:

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

## Example LLM Answer Request

```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do Kubernetes readiness probes work?",
    "method": "hybrid",
    "top_k": 5,
    "generator": "llm"
  }'
```

## Running Evaluations

### Retrieval Evaluation

```bash
PYTHONPATH=. python scripts/run_retrieval_eval.py
```

### Answer Evaluation

Run the default configured evaluation:

```bash
PYTHONPATH=. python scripts/run_answer_eval.py \
  --method hybrid \
  --generator llm
```

Explicitly enable context expansion:

```bash
PYTHONPATH=. python scripts/run_answer_eval.py \
  --method hybrid \
  --generator llm \
  --enable-context-expansion
```

Run only selected questions:

```bash
PYTHONPATH=. python scripts/run_answer_eval.py \
  --method hybrid \
  --generator llm \
  --question-ids q025 q026
```

Run the targeted neighbor-expansion experiment:

```bash
PYTHONPATH=. python scripts/run_answer_eval.py \
  --method hybrid \
  --generator llm \
  --top-k 1 \
  --question-ids q025 q026 \
  --enable-context-expansion
```

### Final Experiment Matrix

Preview the commands without running them:

```bash
PYTHONPATH=. python scripts/run_final_eval_matrix.py \
  --dry-run
```

Run the complete final matrix:

```bash
PYTHONPATH=. python scripts/run_final_eval_matrix.py
```

### Inspect Retrieval Failures

```bash
PYTHONPATH=. python scripts/inspect_retrieval_failures.py
```

### Inspect Neighboring Chunks

```bash
PYTHONPATH=. python scripts/inspect_neighbor_chunks.py \
  --query "ConfigMap" \
  --window 1 \
  --limit 5
```

Experiment outputs are written to:

```text
data/experiments/
```

These generated files are ignored by Git.

## Testing

Run the complete test suite:

```bash
PYTHONPATH=. python -m pytest tests -v
```

The tests cover:

* configuration validation
* BM25 token normalization
* query classification
* routing and abstention
* retrieval metrics
* answer metrics
* citation extraction and evaluation
* context selection
* neighbor expansion behavior
* expansion overhead metrics
* FastAPI answer and search pipelines

The API pipeline tests isolate the database and LLM with test doubles, allowing them to run without Docker or external model calls.

## Experiment Documentation

Detailed experiment history:

```text
docs/experiment_log.md
```

Compact result comparison:

```text
docs/experiment_summary.md
```

Retrieval and answer evaluation report:

```text
docs/evaluation_report.md
```

Failure analysis:

```text
docs/failure_analysis.md
```

## Current Status

The core `v0.1` RAG backend is feature-complete.

Completed capabilities include:

* ingestion and chunking
* PostgreSQL and pgvector storage
* dense, BM25, and Hybrid retrieval
* optional cross-encoder reranking
* query classification and early abstention
* simple and LLM answer generators
* grounded inline citations
* citation alignment checks
* context selection
* configurable neighbor expansion
* retrieval and answer evaluation
* multi-section evidence evaluation
* context-overhead measurement
* reproducible final experiment matrix
* API pipeline regression tests

## Limitations and Future Work

The current benchmark is relatively small and focuses on Kubernetes documentation.

Potential future improvements include:

* larger and more diverse technical-document benchmarks
* multi-document and multi-hop questions
* retrieval-confidence calibration
* adaptive context expansion
* query rewriting
* token-level prompt-cost measurement
* stronger ranking models
* RAGAS or model-based answer-quality evaluation
* frontend chat interface
* deployment and observability infrastructure
