# Evaluation Report

This report summarizes the current retrieval-level and answer-level evaluation results for the RAG system.

The current benchmark contains 20 questions across four query types:

- conceptual: 5
- procedural: 5
- constraint: 5
- out_of_domain: 5

Supported questions are expected to retrieve and answer from the Kubernetes DaemonSet documentation. Out-of-domain questions are expected to trigger abstention.

## Retrieval-Level Evaluation

Retrieval evaluation measures whether the expected section appears in the retrieved chunks.

### Overall Results

| Method | Hit@k | Top-1 Accuracy |
|---|---:|---:|
| Dense | 1.000 | 0.867 |
| BM25 | 1.000 | 0.600 |
| Hybrid | 1.000 | 0.867 |

### Query-Type Sliced Results

| Method | Query Type | Hit@k | Top-1 Accuracy |
|---|---|---:|---:|
| Dense | conceptual | 1.000 | 1.000 |
| Dense | procedural | 1.000 | 0.800 |
| Dense | constraint | 1.000 | 0.800 |
| BM25 | conceptual | 1.000 | 0.600 |
| BM25 | procedural | 1.000 | 1.000 |
| BM25 | constraint | 1.000 | 0.200 |
| Hybrid | conceptual | 1.000 | 1.000 |
| Hybrid | procedural | 1.000 | 0.800 |
| Hybrid | constraint | 1.000 | 0.800 |

## Answer-Level Evaluation

Answer-level evaluation measures whether the system answers supported questions, abstains on unsupported questions, and returns citations appropriately.

### Overall Results

| Method | Abstention Accuracy | Citation Presence Accuracy | Pass Rate |
|---|---:|---:|---:|
| Dense | 1.000 | 1.000 | 1.000 |
| BM25 | 0.850 | 0.850 | 0.850 |
| Hybrid | 1.000 | 1.000 | 1.000 |

### Query-Type Sliced Results

| Method | Query Type | Abstention Accuracy | Citation Accuracy | Pass Rate |
|---|---|---:|---:|---:|
| Dense | conceptual | 1.000 | 1.000 | 1.000 |
| Dense | procedural | 1.000 | 1.000 | 1.000 |
| Dense | constraint | 1.000 | 1.000 | 1.000 |
| Dense | out_of_domain | 1.000 | 1.000 | 1.000 |
| BM25 | conceptual | 0.800 | 0.800 | 0.800 |
| BM25 | procedural | 1.000 | 1.000 | 1.000 |
| BM25 | constraint | 1.000 | 1.000 | 1.000 |
| BM25 | out_of_domain | 0.600 | 0.600 | 0.600 |
| Hybrid | conceptual | 1.000 | 1.000 | 1.000 |
| Hybrid | procedural | 1.000 | 1.000 | 1.000 |
| Hybrid | constraint | 1.000 | 1.000 | 1.000 |
| Hybrid | out_of_domain | 1.000 | 1.000 | 1.000 |

## Key Observations

Dense retrieval performs strongly on the current benchmark, achieving perfect Hit@k and high Top-1 accuracy.

BM25 achieves perfect Hit@k but lower Top-1 accuracy. It performs especially well on procedural queries, where exact terms such as "use", "log collection", and "monitoring" help lexical retrieval. However, it performs poorly on constraint queries, where the relevant section may require more semantic matching.

Hybrid retrieval currently matches dense retrieval on Top-1 accuracy and achieves perfect answer-level pass rate. The evidence sufficiency check improved hybrid behavior on out-of-domain queries.

## Failure Modes

### BM25 Conceptual Query Failure

BM25 sometimes underperforms on short conceptual queries because lexical evidence is sparse. Short questions such as "What is a DaemonSet?" may not produce enough lexical signal after tokenization and scoring.

### BM25 Out-of-Domain False Positives

BM25 can still return results for unrelated queries if there is weak lexical overlap with common terms. This motivates improved tokenization, stopword filtering, and minimum lexical evidence thresholds.

### Hybrid Forced Ranking Risk

Hybrid retrieval originally produced false positives on out-of-domain questions because dense and BM25 retrievers are forced-ranking systems. Even irrelevant queries still receive top-ranked chunks. This was mitigated by adding an evidence sufficiency check based on hybrid score and lexical overlap.

## Next Improvements

- Expand from demo DaemonSet data to real Kubernetes documentation.
- Improve BM25 tokenization with stopword filtering and better normalization.
- Add a query classifier for out-of-domain detection.
- Add experiment comparison reports across retrieval configurations.
- Add LLM-based grounded answer generation.
- Integrate RAGAS or another answer-quality evaluation framework.