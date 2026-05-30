# Failure Analysis

This document tracks retrieval and answer-level failures identified during evaluation, along with the fixes applied.

## Case 1: Hybrid Retriever False Positive on Out-of-Domain Query

### Query

```text
How do I bake a chocolate cake?
```

### Expected Behavior

The system should abstain because the query is unrelated to the Kubernetes DaemonSet documentation.

### Observed Behavior

The hybrid retriever returned DaemonSet-related chunks and the answer generator produced an unsupported answer with citations.

### Root Cause

The hybrid retriever used Reciprocal Rank Fusion (RRF). Since both dense retrieval and BM25 are forced-ranking methods, they can still return a top-ranked chunk even when the query is out of domain.

This created an artificially high fused score, even though the retrieved evidence did not actually support the query.

### Fix

Added an evidence sufficiency check for hybrid retrieval responses:

- Require a minimum hybrid RRF score.
- Require lexical overlap between the query and retrieved evidence.
- Abstain when both retrieval confidence and lexical support are insufficient.

### Result

After the fix, the answer-level evaluation correctly abstained on the out-of-domain query.

### Lesson

Retrievers always return the nearest available documents, but nearest does not necessarily mean relevant. Production RAG systems need evidence sufficiency checks before generation.