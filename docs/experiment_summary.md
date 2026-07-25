# Experiment Summary

This document summarizes the major retrieval, generation, routing, and evaluation experiments for the RAG system.

The goal is to track which changes improved the system, which changes were rejected, and which components remain experimental.

## Current Default Pipeline

The current default pipeline is:

```text
query
→ lightweight query classification
→ early abstention for out-of-domain queries
→ hybrid retrieval
→ context selection
→ grounded LLM answer generation
→ citation alignment
→ answer evaluation
```

The current default retrieval method is:

```text
hybrid
```

The current default answer generator is:

```text
llm
```

The cross-encoder reranker is available as an optional experimental retriever, but it is not the default.

## Current Best Results

### Retrieval-Level Metrics

| Method          | Hit@k | Top-1 Accuracy | Status                          |
| --------------- | ----: | -------------: | ------------------------------- |
| Dense           | 0.778 |          0.389 | Baseline semantic retriever     |
| BM25            | 0.556 |          0.278 | Lexical retriever               |
| Hybrid          | 0.722 |          0.500 | Default retriever               |
| Hybrid Reranked | 0.889 |          0.444 | Optional high-recall experiment |

### Answer-Level Metrics

| Method          | Generator | Abstention Accuracy | Citation Presence Accuracy | Pass Rate | Citation ID Validity | Citation Alignment |
| --------------- | --------- | ------------------: | -------------------------: | --------: | -------------------: | -----------------: |
| Hybrid          | LLM       |               1.000 |                      1.000 |     1.000 |                1.000 |              1.000 |
| Hybrid Reranked | LLM       |               1.000 |                      1.000 |     1.000 |                1.000 |              1.000 |

### Query Classification Metrics

| Metric                    | Score |
| ------------------------- | ----: |
| Accuracy                  | 1.000 |
| Correct count             |    24 |
| Incorrect count           |     0 |
| False in-domain count     |     0 |
| False out-of-domain count |     0 |
| Ambiguous count           |     0 |

## Experiment Decision Table

| Experiment                             | Goal                                                                      | Main Result                                                                          | Decision                    |
| -------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | --------------------------- |
| Metadata-enriched chunk representation | Improve dense retrieval by embedding document and section metadata        | Dense Hit@k improved from 0.611 to 0.778; Top-1 improved from 0.167 to 0.389         | Kept                        |
| Heuristic reranking                    | Improve Top-1 ranking with lexical overlap features                       | Increased failures for dense and hybrid retrieval                                    | Rejected                    |
| Multi-accepted-section evaluation      | Make evaluation fairer when multiple sections can support an answer       | Reduced some false failures but did not eliminate ranking issues                     | Kept                        |
| Field-weighted BM25                    | Improve BM25 by repeating important fields                                | No measurable improvement                                                            | Rejected                    |
| BM25 token normalization               | Improve lexical matching for Kubernetes-specific plural and variant terms | BM25 Hit@k improved from 0.389 to 0.556; wrong-document failures dropped from 7 to 1 | Kept                        |
| LLM grounded answer generation         | Generate natural answers from retrieved context                           | Full 24-question answer evaluation achieved 1.000 pass rate with hybrid retrieval    | Kept                        |
| Citation alignment                     | Return only citations actually referenced in generated answers            | Citation validity and alignment reached 1.000                                        | Kept                        |
| Query classification                   | Separate in-domain and out-of-domain queries                              | Classification accuracy reached 1.000 on the benchmark                               | Kept                        |
| Early abstention routing               | Avoid retrieval and generation for out-of-domain queries                  | Out-of-domain queries abstain before retrieval or LLM generation                     | Kept                        |
| Context selection                      | Remove boilerplate sections before LLM generation                         | Reduces prompt noise from sections such as `Feedback` and `What's next`              | Kept                        |
| Cross-encoder reranking                | Improve section-level ranking with second-stage reranking                 | Hit@k improved from 0.722 to 0.889, but Top-1 dropped from 0.500 to 0.444            | Kept as optional experiment |

## Key Lessons

### 1. Retrieval recall and ranking quality are different

The cross-encoder reranker improved Hit@k but reduced Top-1 accuracy. This shows that retrieving the correct evidence somewhere in the candidate set is not the same as ranking it first.

### 2. Lexical normalization matters for technical documentation

BM25 performed poorly when Kubernetes-specific terms such as `services`, `secrets`, `deployments`, and `pods` were treated as unrelated variants. Lightweight token normalization significantly improved BM25 document routing.

### 3. Simple heuristic reranking can hurt retrieval

The heuristic reranker was rejected because it increased failures. This reinforced the need to evaluate retrieval changes instead of assuming that additional ranking logic improves performance.

### 4. Citation alignment should be enforced in code

Prompt instructions alone are not enough. The system now extracts citation IDs from generated answers and returns only citations that are explicitly referenced.

### 5. Answer-level evaluation can hide retrieval differences

Both hybrid and hybrid-reranked retrieval achieved perfect answer-level results with the LLM generator, even though their retrieval-level metrics differed. This means retrieval metrics and answer metrics should both be tracked.

### 6. Context quality matters before generation

The context selector removes generic documentation boilerplate such as `Feedback` and `What's next` before LLM generation. This reduces prompt noise and helps keep answer generation focused on useful evidence.

### 7. Query classification is useful as a routing layer

The lightweight classifier correctly separated in-domain and out-of-domain queries on the current benchmark. This enables early abstention and avoids unnecessary retrieval or LLM calls for clearly unsupported queries.

## Current Open Problems

The main remaining retrieval issue is section-level ranking.

Common remaining failure types include:

* right document but wrong section
* expected section appears in top-k but not top-1
* generic documentation sections appearing in retrieved context
* retrieval recall and answer-level quality not always moving together

## Potential Next Improvements

Potential next improvements include:

* stronger context selection
* parent-child retrieval
* section-neighbor expansion
* query-aware context compression
* LLM-based groundedness evaluation
* RAGAS integration
* integration tests for the retrieval and answer-generation pipeline

## Current Project Status

The project now supports an evaluated end-to-end RAG workflow:

```text
real technical documentation
→ document ingestion
→ chunking and metadata preservation
→ dense, BM25, and hybrid retrieval
→ optional cross-encoder reranking
→ query classification
→ early abstention
→ context selection
→ grounded LLM answer generation
→ citation alignment
→ answer-level and retrieval-level evaluation
```

The current default system uses hybrid retrieval with LLM-based grounded answer generation. The optional cross-encoder reranker is retained for future high-recall retrieval experiments but is not currently the default strategy.

## Final Answer-Evaluation Results

The final answer-evaluation benchmark contains:

| Dataset property                 | Count |
| -------------------------------- | ----: |
| Total questions                  |    26 |
| Supported questions              |    20 |
| Unsupported questions            |     6 |
| Required multi-section questions |     2 |

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

The overall citation-utilization result is `0.769` because 20 of the 26 questions were supported and returned citations. The six unsupported questions correctly abstained without citations. Among answered questions, citation utilization was `1.000`.

## Final Retrieval and Generation Decisions

### Default Retriever

Hybrid retrieval remains the default strategy.

It provides the best balance of lexical document routing and semantic section retrieval. Cross-encoder reranking remains available as an optional experimental strategy, but it did not improve final answer-level performance on the current benchmark.

### Context Expansion

Section-neighbor expansion is retained as a configurable feature.

The final experiments showed two different behaviors:

* With `top_k=5`, expansion did not improve answer-level metrics but added an average of three chunks and approximately 2192 characters per retrieval question.
* With `top_k=1`, expansion increased required evidence coverage from `0.500` to `1.000` on two targeted neighbor-dependent questions.

This demonstrates that expansion is useful when initial retrieval is narrow or when evidence spans adjacent chunks, but it is not justified as an unconditional operation under the current default retrieval configuration.

The recommended policy is to keep expansion disabled by default for `top_k=5` and enable it explicitly for constrained-retrieval, multi-section, or low-confidence workflows.

### Final Pipeline

The completed evaluated pipeline is:

`query classification → early abstention → hybrid retrieval → optional reranking → optional neighbor expansion → context selection → grounded LLM generation → citation filtering → answer and evidence evaluation`

The system supports:

* grounded answer generation from retrieved documentation
* safe abstention for unsupported questions
* inline citation validation and alignment
* required multi-section evidence coverage
* configurable retrieval and generation behavior
* reproducible experiment matrices
* measurement of context-expansion quality and overhead

## Current Limitations

The current benchmark is relatively small and is focused on Kubernetes documentation.

Answer-level performance is saturated under the default Hybrid configuration, so aggregate pass rate alone is not sufficient to distinguish every retrieval or context-processing strategy.

Future evaluation should include:

* more difficult multi-document questions
* more neighbor-dependent questions
* paraphrased and ambiguous queries
* noisy or partially relevant retrieved context
* retrieval-confidence calibration
* adaptive rather than unconditional context expansion
* token-level prompt-cost measurement
