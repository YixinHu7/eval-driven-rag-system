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