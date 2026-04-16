from src.ingestion.parser import parse_markdown_document
from src.ingestion.chunker import split_markdown_by_headings
from src.storage.db import SessionLocal
from src.storage.repository import DocumentRepository, ChunkRepository


def main() -> None:
    file_path = "data/raw/daemonset_demo.md"
    doc_id = "k8s_daemonset_demo"

    document = parse_markdown_document(
        file_path=file_path,
        doc_id=doc_id,
        source_url="https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/",
        category="workloads",
    )

    chunks = split_markdown_by_headings(doc_id=document.doc_id, text=document.raw_text)

    with SessionLocal() as session:
        document_repo = DocumentRepository(session)
        chunk_repo = ChunkRepository(session)

        inserted_doc = document_repo.insert_document(document)
        inserted_chunks = chunk_repo.insert_chunks(chunks)

        print(f"Inserted document: {inserted_doc.doc_id}")
        print(f"Inserted chunks: {len(inserted_chunks)}")

        for chunk in inserted_chunks:
            print(
                f"- {chunk.chunk_id} | section={chunk.section_title} | tokens={chunk.token_count}"
            )


if __name__ == "__main__":
    main()