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

## Experiment 8: Context Selection Before LLM Generation

### Hypothesis

The LLM answer generator may produce cleaner grounded answers if generic documentation boilerplate is removed before prompt construction.

Real documentation often contains low-value sections such as:

* `Feedback`
* `What's next`

These sections may be retrieved because they appear near relevant documentation, but they usually do not provide direct evidence for user questions.

### Change

Added a context selection layer before LLM answer generation.

The context selector filters out obvious non-evidence chunks, including sections such as:

* `Feedback`
* `What's next`

The selector intentionally does not remove sections such as `Caution:` because caution sections can contain important technical evidence in Kubernetes documentation.

### Result

The context selector reduced prompt noise before LLM generation while preserving answer correctness.

The full hybrid LLM answer evaluation remained stable:

| Metric                     | Result |
| -------------------------- | -----: |
| Abstention accuracy        |  1.000 |
| Citation presence accuracy |  1.000 |
| Pass rate                  |  1.000 |
| Citation ID validity rate  |  1.000 |
| Citation alignment rate    |  1.000 |

### Conclusion

Context selection was kept because it improves evidence quality before generation without reducing answer-level performance.

This change supports a cleaner generation pipeline:

`retrieved chunks → context selection → grounded prompt → LLM answer → citation alignment`

## Experiment 9: Section-Neighbor Context Expansion

### Hypothesis

Some technical-documentation questions require evidence from adjacent chunks or nearby sections rather than from only the highest-ranked retrieved chunk.

Section-neighbor context expansion may improve evidence completeness by adding nearby chunks from the same document. However, unconditional expansion may also increase prompt size without improving answer quality when the initial retriever already returns sufficient context.

### Change

Added configurable section-neighbor context expansion before answer generation.

The expansion pipeline:

1. Starts with the initial retrieved chunks.
2. Preserves all initial seed chunks before adding neighbors.
3. Finds adjacent chunks from the same document using the chunk index.
4. Deduplicates repeated chunks.
5. Limits expansion using:

   * neighbor window
   * maximum final chunk count
   * maximum added neighbor-context characters
6. Reassigns stable sequential ranks.
7. Passes the resulting context through context selection before generation.

The evaluated configuration used:

| Setting                             | Value |
| ----------------------------------- | ----: |
| Expansion window                    |     1 |
| Maximum expanded chunks             |     8 |
| Maximum neighbor-context characters |  6000 |

The answer evaluator was also extended with `required_evidence_sections`, allowing multi-section questions to pass only when the answer actually cites every required evidence section.

Two targeted multi-section questions were added:

* a StatefulSet question requiring both `Pod Identity` and `Ordinal Index`
* a Deployment question requiring both `Rolling Update Deployment` and `Max Unavailable`

### Final Evaluation Matrix

Five final scenarios were evaluated using the LLM answer generator.

| Scenario                                      | Questions | Pass Rate | Required Evidence Accuracy | Evidence Coverage | Average Added Neighbors | Average Added Characters |
| --------------------------------------------- | --------: | --------: | -------------------------: | ----------------: | ----------------------: | -----------------------: |
| Hybrid, top-k 5, expansion enabled            |        26 |     1.000 |                      1.000 |             1.000 |                   3.000 |                  2191.95 |
| Hybrid, top-k 5, expansion disabled           |        26 |     1.000 |                      1.000 |             1.000 |                   0.000 |                     0.00 |
| Hybrid, top-k 1, targeted, expansion enabled  |         2 |     1.000 |                      1.000 |             1.000 |                   2.000 |                  1543.50 |
| Hybrid, top-k 1, targeted, expansion disabled |         2 |     0.000 |                      0.000 |             0.500 |                   0.000 |                     0.00 |
| Hybrid reranked, top-k 5, expansion enabled   |        26 |     1.000 |                      1.000 |             1.000 |                   3.000 |                  2007.85 |

All scenarios maintained:

* citation ID validity rate: `1.000`
* citation alignment rate: `1.000`
* answered-only citation utilization: `1.000`

For the full 26-question evaluations, average citation utilization was `0.769`. This reflects the 20 supported questions that returned citations and the 6 unsupported questions that correctly abstained without citations.

### Default Top-k Result

With the default initial retrieval setting of `top_k=5`, both expansion-enabled and expansion-disabled Hybrid evaluation achieved:

* abstention accuracy: `1.000`
* citation presence accuracy: `1.000`
* pass rate: `1.000`
* required evidence accuracy: `1.000`
* average required evidence coverage: `1.000`

Expansion did not provide measurable answer-quality improvement under this configuration.

It did, however, add neighbors for every retrieval question:

* questions with added neighbors: `20`
* neighbor addition rate: `1.000`
* average initial chunks: `5.000`
* average final chunks: `8.000`
* average added neighbors: `3.000`
* average added context characters: `2191.95`

The expansion process therefore reached the maximum final chunk count for every question that performed retrieval.

### Targeted Top-k 1 Ablation

A stricter experiment used `top_k=1` and evaluated only the two neighbor-dependent multi-section questions.

With expansion enabled:

* pass rate: `1.000`
* required evidence accuracy: `1.000`
* average required evidence coverage: `1.000`
* average initial chunks: `1.000`
* average final chunks: `3.000`
* average added neighbors: `2.000`
* average added context characters: `1543.50`

With expansion disabled:

* pass rate: `0.000`
* required evidence accuracy: `0.000`
* average required evidence coverage: `0.500`

Each disabled-expansion answer cited only one of the two required evidence sections. Expansion successfully added the adjacent section and allowed both targeted questions to pass.

### Conclusion

Section-neighbor context expansion provides measurable value when initial retrieval is narrow and the answer requires evidence from adjacent chunks.

However, unconditional expansion is not cost-effective under the current default `top_k=5` configuration. It added an average of three chunks and approximately 2192 characters to every retrieval question without improving answer correctness, citation quality, or required evidence coverage.

The final design decision is:

* retain section-neighbor expansion as a configurable capability
* preserve its character and chunk-count limits
* use it for constrained-retrieval or neighbor-dependent workflows
* avoid treating it as an unconditional default when `top_k=5`
* consider adaptive expansion as future work

An adaptive policy could trigger expansion only when retrieval confidence is low, evidence requirements are incomplete, or the initial context indicates that an adjacent section is needed.
