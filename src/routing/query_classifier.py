import re
from typing import Literal

from pydantic import BaseModel


QueryDomain = Literal["in_domain", "out_of_domain", "ambiguous"]


KUBERNETES_TERMS = {
    "kubernetes",
    "k8s",
    "pod",
    "pods",
    "container",
    "containers",
    "service",
    "services",
    "daemonset",
    "daemonsets",
    "deployment",
    "deployments",
    "statefulset",
    "statefulsets",
    "ingress",
    "configmap",
    "configmaps",
    "secret",
    "secrets",
    "probe",
    "probes",
    "liveness",
    "readiness",
    "startup",
    "kubelet",
    "node",
    "nodes",
    "cluster",
    "namespace",
    "namespaces",
    "yaml",
    "manifest",
    "replica",
    "replicas",
    "selector",
    "selectors",
    "label",
    "labels",
    "traffic",
    "rollout",
    "rollouts",
    "environment",
    "variable",
    "variables",
}


class QueryClassification(BaseModel):
    query_type: QueryDomain
    confidence: float
    matched_terms: list[str]


def tokenize_query(query: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", query.lower())


def classify_query(query: str) -> QueryClassification:
    tokens = tokenize_query(query)
    token_set = set(tokens)

    matched_terms = sorted(token_set & KUBERNETES_TERMS)

    if matched_terms:
        return QueryClassification(
            query_type="in_domain",
            confidence=min(1.0, 0.5 + 0.1 * len(matched_terms)),
            matched_terms=matched_terms,
        )

    if len(tokens) <= 2:
        return QueryClassification(
            query_type="ambiguous",
            confidence=0.3,
            matched_terms=[],
        )

    return QueryClassification(
        query_type="out_of_domain",
        confidence=0.7,
        matched_terms=[],
    )