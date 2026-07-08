from src.routing.query_classifier import classify_query, tokenize_query


def test_tokenize_query_lowercases_and_removes_punctuation() -> None:
    tokens = tokenize_query("How do Kubernetes readiness probes work?")

    assert "kubernetes" in tokens
    assert "readiness" in tokens
    assert "probes" in tokens


def test_classifies_kubernetes_query_as_in_domain() -> None:
    result = classify_query("How do Kubernetes readiness probes work?")

    assert result.query_type == "in_domain"
    assert result.confidence > 0.0
    assert "kubernetes" in result.matched_terms
    assert "readiness" in result.matched_terms
    assert "probes" in result.matched_terms


def test_classifies_unrelated_query_as_out_of_domain() -> None:
    result = classify_query("How do I bake a chocolate cake?")

    assert result.query_type == "out_of_domain"
    assert result.matched_terms == []


def test_classifies_short_generic_query_as_ambiguous() -> None:
    result = classify_query("Explain it")

    assert result.query_type == "ambiguous"
    assert result.matched_terms == []