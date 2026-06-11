from src.retrieval.bm25 import normalize_token, simple_tokenize


def test_normalize_kubernetes_plural_terms() -> None:
    assert normalize_token("services") == "service"
    assert normalize_token("secrets") == "secret"
    assert normalize_token("configmaps") == "configmap"
    assert normalize_token("deployments") == "deployment"
    assert normalize_token("pods") == "pod"
    assert normalize_token("nodes") == "node"


def test_simple_tokenize_removes_stopwords_and_normalizes_terms() -> None:
    tokens = simple_tokenize("How do I expose Services with Pods?")

    assert "how" not in tokens
    assert "do" not in tokens
    assert "i" not in tokens
    assert "service" in tokens
    assert "pod" in tokens