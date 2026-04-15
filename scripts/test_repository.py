from src.core.models import ChunkRecord, DocumentRecord
from src.storage.db import SessionLocal
from src.storage.repository import ChunkRepository, DocumentRepository


def main() -> None:
    with SessionLocal() as session:
        document_repo = DocumentRepository(session)
        chunk_repo = ChunkRepository(session)

        doc = DocumentRecord(
            doc_id="doc_demo_001",
            title="Kubernetes DaemonSet Overview",
            source_url="https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/",
            category="workloads",
            section_path="Workloads > DaemonSet",
            parent_section="Workloads",
            version=None,
            last_updated=None,
            raw_text=(
                "A DaemonSet ensures that all or some Nodes run a copy of a Pod. "
                "As nodes are added to the cluster, Pods are added to them."
            ),
        )

        inserted_doc = document_repo.insert_document(doc)
        print(f"Inserted document: {inserted_doc.doc_id}")

        chunks = [
            ChunkRecord(
                chunk_id="chunk_demo_001",
                doc_id="doc_demo_001",
                chunk_index=0,
                chunk_text="A DaemonSet ensures that all or some Nodes run a copy of a Pod.",
                section_title="DaemonSet",
                section_path="Workloads > DaemonSet",
                token_count=13,
                embedding=None,
            ),
            ChunkRecord(
                chunk_id="chunk_demo_002",
                doc_id="doc_demo_001",
                chunk_index=1,
                chunk_text="As nodes are added to the cluster, Pods are added to them.",
                section_title="DaemonSet",
                section_path="Workloads > DaemonSet",
                token_count=12,
                embedding=None,
            ),
        ]

        inserted_chunks = chunk_repo.insert_chunks(chunks)
        print(f"Inserted chunks: {len(inserted_chunks)}")

        loaded_doc = document_repo.get_document_by_id("doc_demo_001")
        print(f"Loaded document title: {loaded_doc.title if loaded_doc else 'None'}")

        loaded_chunks = chunk_repo.list_chunks_for_doc("doc_demo_001")
        print(f"Loaded chunks for doc_demo_001: {len(loaded_chunks)}")

        print(f"Document count: {document_repo.count_documents()}")
        print(f"Chunk count: {chunk_repo.count_chunks()}")


if __name__ == "__main__":
    main()