# Experiment Log

This document tracks retrieval and evaluation experiments, including the hypothesis, change, metrics, and conclusion for each iteration.

## Experiment 1: Metadata-Enriched Chunk Representation

### Hypothesis

Dense retrieval may perform better if chunk embeddings include document and section metadata, not only raw chunk text.

### Change

Updated the embedding text representation to include:

- document title
- category
- section path
- section title
- chunk content

### Result

Dense retrieval improved:

| Metric | Before | After |
|---|---:|---:|
| Hit@k | 0.611 | 0.778 |
| Top-1 Accuracy | 0.167 | 0.389 |

### Conclusion

Metadata-enriched chunk representation improved section-level dense retrieval. This suggests that document and section context are important signals for technical documentation retrieval.

## Experiment 2: Heuristic Reranking

### Hypothesis

A lightweight lexical-overlap reranker may improve Top-1 accuracy by promoting chunks whose section titles and content overlap with the query.

### Change

Tested a heuristic reranker using overlap between query tokens and:

- section title
- section path
- document title
- content

### Result

The heuristic reranker did not improve retrieval performance.

| Method | Before Failures | After Failures |
|---|---:|---:|
| Dense | 11 | 14 |
| Hybrid | 12 | 13 |

### Conclusion

The heuristic reranker was not retained. Simple lexical-overlap reranking can hurt semantic retrieval when section titles are short, generic, or weakly aligned with natural-language queries.

## Experiment 3: Multi-Accepted-Section Evaluation

### Hypothesis

Some questions may be correctly answered from multiple reasonable sections, so single-section evaluation may be too strict.

### Change

Added support for `accepted_doc_sections` in the evaluation dataset.

### Result

Failure counts decreased slightly:

| Failure Type | Before | After |
|---|---:|---:|
| expected_section_in_top_k_but_not_top_1 | 15 | 14 |
| right_document_wrong_section | 14 | 12 |
| wrong_document | 8 | 8 |

### Conclusion

Multi-accepted-section evaluation made the benchmark more realistic, but most failures remained true retrieval or ranking issues.

## Experiment 4: Field-Weighted BM25

### Hypothesis

BM25 may improve if important fields such as section title and document title are repeated to simulate higher field weights.

### Change

Tested field-weighted BM25 text by repeating section titles, document titles, and section paths.

### Result

No measurable improvement was observed.

### Conclusion

Field weighting alone did not improve BM25. The issue was likely more related to token normalization than field weighting.

## Experiment 5: BM25 Token Normalization

### Hypothesis

BM25 may fail on technical documentation because plural and variant terms are treated as different tokens, such as `services` vs `service` and `secrets` vs `secret`.

### Change

Added lightweight token normalization for Kubernetes-related technical terms, including:

- services → service
- secrets → secret
- configmaps → configmap
- deployments → deployment
- pods → pod
- nodes → node
- probes → probe

### Result

BM25 improved substantially:

| Metric | Before | After |
|---|---:|---:|
| Hit@k | 0.389 | 0.556 |
| Top-1 Accuracy | 0.222 | 0.278 |
| Wrong-document failures | 7 | 1 |

Hybrid retrieval also improved because its BM25 component became cleaner:

| Metric | Before | After |
|---|---:|---:|
| Hit@k | 0.611 | 0.722 |
| Top-1 Accuracy | 0.333 | 0.500 |
| Wrong-document failures | 1 | 0 |

### Conclusion

BM25 token normalization improved lexical document routing. Hybrid retrieval benefited from the cleaner BM25 signal. Weighted hybrid fusion was tested separately and did not add measurable improvement beyond BM25 normalization.

## Experiment 6: LLM Grounded Answer Generation

### Hypothesis

A grounded LLM answer generator can improve the user-facing answer quality while preserving evaluation reliability if it is constrained to retrieved context and citation alignment is enforced in code.

### Change

Added an LLM-based answer generator with the following behavior:

* uses retrieved chunks as the only answer context
* requires inline citation markers such as `[1]`
* abstains before generation when retrieved evidence is insufficient
* filters returned citations to include only citations explicitly referenced in the answer
* fails safely if the generated answer contains no usable citations

The answer evaluation pipeline was updated to support configurable generators:

* `simple`
* `llm`

Citation evaluation was also added to measure:

* citation ID validity
* citation alignment
* citation utilization
* answered-only citation utilization

### Result

The LLM generator was evaluated on the full 24-question Kubernetes documentation benchmark using hybrid retrieval.

| Metric                                | Result |
| ------------------------------------- | -----: |
| Total questions                       |     24 |
| Supported questions                   |     18 |
| Abstention accuracy                   |  1.000 |
| Citation presence accuracy            |  1.000 |
| Pass rate                             |  1.000 |
| Citation ID validity rate             |  1.000 |
| Citation alignment rate               |  1.000 |
| Average citation utilization          |  0.750 |
| Average answered citation utilization |  1.000 |

### Conclusion

The LLM generator successfully completed the end-to-end grounded answer workflow. It answered supported questions with aligned citations and abstained on unsupported questions.

The `average_citation_utilization` score is 0.750 because unsupported questions correctly returned no citations. For answered questions only, citation utilization was 1.000.

This result validates the project’s core RAG loop:

`retrieve → generate → cite → abstain → evaluate`

## Experiment 7: Cross-Encoder Reranking

### Hypothesis

A cross-encoder reranker may improve section-level ranking by scoring query-chunk pairs more precisely than dense retrieval or BM25 alone.

### Change

Added optional second-stage reranked retrievers:

* `dense_reranked`
* `hybrid_reranked`

The reranker first retrieves candidate chunks using a base retriever, then reorders those candidates using a cross-encoder model.

### Retrieval-Level Result

The reranked hybrid retriever improved Hit@k but did not improve Top-1 accuracy.

| Method          | Hit@k | Top-1 Accuracy |
| --------------- | ----: | -------------: |
| Hybrid          | 0.722 |          0.500 |
| Hybrid Reranked | 0.889 |          0.444 |

### Answer-Level Result

Both hybrid and hybrid-reranked retrieval achieved perfect answer-level performance with the LLM answer generator on the current benchmark.

| Method          | Generator | Abstention Accuracy | Citation Presence Accuracy | Pass Rate | Citation ID Validity | Citation Alignment |
| --------------- | --------- | ------------------: | -------------------------: | --------: | -------------------: | -----------------: |
| Hybrid          | LLM       |               1.000 |                      1.000 |     1.000 |                1.000 |              1.000 |
| Hybrid Reranked | LLM       |               1.000 |                      1.000 |     1.000 |                1.000 |              1.000 |

### Conclusion

Cross-encoder reranking increased the likelihood that the correct evidence section appears somewhere in the top-k results, but it did not improve Top-1 ranking on the current benchmark.

Because answer-level performance was already perfect with the default hybrid retriever, the reranked retriever did not provide measurable end-to-end improvement in this evaluation.

The reranker is kept as an optional experimental retriever for high-recall retrieval experiments, but hybrid retrieval remains the default strategy.
