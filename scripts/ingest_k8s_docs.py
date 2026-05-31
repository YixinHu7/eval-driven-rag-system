import hashlib
import time
from dataclasses import dataclass

from src.core.models import DocumentRecord
from src.ingestion.chunker import split_markdown_by_headings
from src.ingestion.crawler import fetch_web_document
from src.storage.db import SessionLocal
from src.storage.repository import ChunkRepository, DocumentRepository


@dataclass(frozen=True)
class K8sDocSource:
    doc_id: str
    url: str
    category: str
    parent_section: str


K8S_DOC_SOURCES = [
    K8sDocSource(
        doc_id="k8s_daemonset",
        url="https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/",
        category="workloads",
        parent_section="Workloads",
    ),
    K8sDocSource(
        doc_id="k8s_deployment",
        url="https://kubernetes.io/docs/concepts/workloads/controllers/deployment/",
        category="workloads",
        parent_section="Workloads",
    ),
    K8sDocSource(
        doc_id="k8s_statefulset",
        url="https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/",
        category="workloads",
        parent_section="Workloads",
    ),
    K8sDocSource(
        doc_id="k8s_service",
        url="https://kubernetes.io/docs/concepts/services-networking/service/",
        category="services-networking",
        parent_section="Services, Load Balancing, and Networking",
    ),
    K8sDocSource(
        doc_id="k8s_ingress",
        url="https://kubernetes.io/docs/concepts/services-networking/ingress/",
        category="services-networking",
        parent_section="Services, Load Balancing, and Networking",
    ),
    K8sDocSource(
        doc_id="k8s_configmap",
        url="https://kubernetes.io/docs/concepts/configuration/configmap/",
        category="configuration",
        parent_section="Configuration",
    ),
    K8sDocSource(
        doc_id="k8s_secret",
        url="https://kubernetes.io/docs/concepts/configuration/secret/",
        category="configuration",
        parent_section="Configuration",
    ),
    K8sDocSource(
        doc_id="k8s_probes",
        url="https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/",
        category="tasks",
        parent_section="Configure Pods and Containers",
    ),
]


def stable_chunk_id(doc_id: str, chunk_index: int, chunk_text: str) -> str:
    digest = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}_chunk_{chunk_index:03d}_{digest}"


def build_document_record(source: K8sDocSource, title: str, text: str) -> DocumentRecord:
    return DocumentRecord(
        doc_id=source.doc_id,
        title=title,
        source_url=source.url,
        category=source.category,
        section_path=f"{source.parent_section} > {title}",
        parent_section=source.parent_section,
        version=None,
        last_updated=None,
        raw_text=text,
    )


def ingest_source(source: K8sDocSource) -> tuple[str, int]:
    web_doc = fetch_web_document(source.url)

    document = build_document_record(
        source=source,
        title=web_doc.title,
        text=web_doc.text,
    )

    chunks = split_markdown_by_headings(
        doc_id=document.doc_id,
        text=document.raw_text,
    )

    # Reassign stable chunk IDs because real docs may change over time.
    normalized_chunks = []
    for idx, chunk in enumerate(chunks):
        normalized_chunks.append(
            chunk.model_copy(
                update={
                    "chunk_id": stable_chunk_id(
                        document.doc_id,
                        idx,
                        chunk.chunk_text,
                    ),
                    "section_path": f"{document.section_path} > {chunk.section_title}",
                }
            )
        )

    with SessionLocal() as session:
        document_repo = DocumentRepository(session)
        chunk_repo = ChunkRepository(session)

        document_repo.insert_document(document)
        chunk_repo.insert_chunks(normalized_chunks)

    return document.doc_id, len(normalized_chunks)


def main() -> None:
    total_chunks = 0

    for source in K8S_DOC_SOURCES:
        print(f"Ingesting {source.doc_id}: {source.url}")
        doc_id, chunk_count = ingest_source(source)
        total_chunks += chunk_count
        print(f"Inserted {doc_id} with {chunk_count} chunks")

        # Be polite to the website and avoid rapid repeated requests.
        time.sleep(0.5)

    print("=" * 80)
    print(f"Ingested {len(K8S_DOC_SOURCES)} documents")
    print(f"Inserted {total_chunks} chunks")


if __name__ == "__main__":
    main()