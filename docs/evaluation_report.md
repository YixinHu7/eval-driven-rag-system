# Evaluation Report

This report summarizes the current retrieval-level and answer-level evaluation results for the RAG system.

The current benchmark contains 24 questions across four query types:

* conceptual
* procedural
* constraint
* out_of_domain

The benchmark is built over real Kubernetes documentation, including topics such as DaemonSets, Deployments, StatefulSets, Services, Ingress, ConfigMaps, Secrets, and liveness/readiness/startup probes.

Supported questions are expected to retrieve relevant Kubernetes documentation sections and produce grounded answers with citations. Out-of-domain questions are expected to trigger abstention.

## Retrieval-Level Evaluation

Retrieval evaluation measures whether the expected document section appears in the retrieved chunks.

The evaluation is document-aware. A retrieved section is considered correct only when both the document ID and the section title match an accepted evidence section. This avoids false positives from repeated generic section names such as `Caution:`, `Note:`, `Feedback`, and `What's next`.

### Overall Results

| Method | Hit@k | Top-1 Accuracy |
| ------ | ----: | -------------: |
| Dense  | 0.778 |          0.389 |
| BM25   | 0.556 |          0.278 |
| Hybrid | 0.722 |          0.500 |

### Interpretation

Dense retrieval achieved the strongest Hit@k, which means it is effective at retrieving relevant evidence somewhere in the top-k results.

BM25 achieved lower overall retrieval performance, but token normalization improved its document-level routing by reducing wrong-document failures. Terms such as `services`, `secrets`, `deployments`, `pods`, and `probes` are normalized to improve lexical matching.

Hybrid retrieval achieved the best Top-1 accuracy. This suggests that BM25 provides useful lexical routing signals, while dense retrieval provides stronger semantic matching.

## Answer-Level Evaluation

Answer-level evaluation measures whether the system answers supported questions, abstains on unsupported questions, and returns citations appropriately.

The current answer-level evaluation uses the LLM-based grounded answer generator with hybrid retrieval.

### Configuration

| Component                             | Setting                                                                   |
| ------------------------------------- | ------------------------------------------------------------------------- |
| Retriever                             | Hybrid retrieval                                                          |
| Generator                             | LLM grounded answer generator                                             |
| Evaluation size                       | 24 questions                                                              |
| Supported questions                   | 18                                                                        |
| Unsupported / out-of-domain questions | 6                                                                         |
| Citation handling                     | Only citations explicitly referenced by the generated answer are returned |
| Abstention behavior                   | Unsupported questions should abstain before generating an answer          |

### Overall Results

| Metric                                | Score |
| ------------------------------------- | ----: |
| Abstention accuracy                   | 1.000 |
| Citation presence accuracy            | 1.000 |
| Pass rate                             | 1.000 |
| Citation ID validity rate             | 1.000 |
| Citation alignment rate               | 1.000 |
| Average citation utilization          | 0.750 |
| Average answered citation utilization | 1.000 |

### Interpretation

The LLM generator successfully answered all supported questions and abstained on all unsupported questions.

The average citation utilization across all 24 questions is 0.750 because the 6 unsupported questions correctly abstained and returned no citations. Among answered questions only, citation utilization was 1.000.

This confirms that the system can:

* retrieve relevant documentation chunks
* generate grounded technical answers
* return only citations used by the generated answer
* abstain on unsupported questions
* preserve citation alignment between answer text and returned citation metadata

## Key Observations

Dense retrieval performs well for broad technical-documentation questions because it captures semantic similarity beyond exact keyword overlap.

BM25 improved after Kubernetes-specific token normalization. The largest improvement was a reduction in wrong-document retrieval failures, showing that lexical normalization is useful for document-level routing.

Hybrid retrieval currently provides the strongest Top-1 retrieval accuracy. It benefits from both semantic retrieval and lexical matching.

The LLM answer generator achieved a perfect answer-level pass rate on the current 24-question benchmark when paired with hybrid retrieval. It correctly answered supported questions, abstained on unsupported questions, and preserved citation alignment.

The project now supports an evaluated end-to-end RAG workflow:

`query → hybrid retrieval → abstention check → grounded LLM answer → citation alignment → answer evaluation`

## Failure Modes

### Right Document but Wrong Section

The most common retrieval failure is retrieving the correct Kubernetes document but ranking the wrong section above the expected evidence section.

This usually happens when multiple sections in the same document contain overlapping terminology. For example, probe-related questions may retrieve sections about liveness probes, readiness probes, caution notes, or examples from the same page.

### Expected Section in Top-k but Not Top-1

Some queries retrieve the expected section within the top-k results but fail to rank it first. This indicates that retrieval recall is often acceptable, but section-level ranking still needs improvement.

### Generic Documentation Sections

Real Kubernetes documentation includes generic sections such as `What's next`, `Feedback`, `Note:`, and `Caution:`. These sections can sometimes appear in retrieved results even when they are not the best evidence for the answer.

### Forced Ranking Risk

Dense, BM25, and hybrid retrieval are forced-ranking systems. Even unrelated queries can return top-ranked chunks. This risk is mitigated by abstention logic that checks evidence sufficiency before answer generation.

## Completed Improvements

The following improvements have already been implemented:

* expanded from demo DaemonSet data to real Kubernetes documentation
* added document-aware retrieval evaluation
* added multi-accepted-section evaluation
* enriched chunk representations with document and section metadata
* added BM25 token normalization for Kubernetes-specific terms
* added retrieval failure inspection reports
* added LLM-based grounded answer generation
* added citation alignment between generated answers and returned citation metadata
* added configurable answer evaluation for simple and LLM generators
* added citation validity and citation alignment metrics

## Next Improvements

* Improve section-level ranking for right-document-wrong-section failures.
* Add query classification for stronger out-of-domain and unsupported-query detection.
* Add optional cross-encoder reranking over hybrid candidates.
* Improve context selection before LLM answer generation.
* Add LLM-based groundedness evaluation.
* Integrate RAGAS or another answer-quality evaluation framework.
* Add integration tests for the retrieval and answer-generation pipeline.
