from pathlib import Path
from src.vectorstore import ensure_pinecone_index
from src.ingestion import ingest_dir, SUPPORTED
from src.config import get_settings


root = Path(__file__).resolve().parent  
document_dir = root /"documents"

def main():
    s = get_settings()

    if not s.GEMINI_API_KEY:
        raise RuntimeError("GEMINi_API_KEY is missing. Add it to your .env file.")
    if not s.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is missing. Add it to your .env file.")

    print("=" * 68)
    print("CloudOps Sentinel - Pinecone Knowledge Base Ingestion")
    print("=" * 68)
    print(f"gemini embedding model : {s.GEMINI_EMBEDDING_MODEL}")
    print(f"Embedding dimension    : {s.EMBEDDING_DIM}")
    print(f"Pinecone index         : {s.PINECONE_INDEX_NAME}")
    print(f"Pinecone namespace     : {s.PINECONE_NAMESPACE}")
    print(f"Documents directory    : {document_dir}")
    print()

    ensure_pinecone_index()
    print("[1/2] Pinecone index is ready.")

    files = [
        p for p in sorted(document_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED
    ] if document_dir.exists() else []

    if not files:
        print("[2/2] No supported documents found. Add files to ./documents and run again.")
        return

    print(f"[2/2] Ingesting {len(files)} document(s)...")
    for path in files:
        print(f"      - {path.name}")

    total_chunks = ingest_dir(document_dir)

    print()
    print("Knowledge base is ready.")
    print(f"Indexed chunks: {total_chunks}")
    print(f"Index: {s.PINECONE_INDEX_NAME}")
    print(f"Namespace: {s.PINECONE_NAMESPACE}")


if __name__ == "__main__":
    main()