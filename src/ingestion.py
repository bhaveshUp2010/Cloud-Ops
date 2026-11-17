from pathlib import Path
from hashlib import sha256
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document as docxDocument
from src.config import get_settings
from src.vectorstore import get_vector_store

SUPPORTED = {".txt", ".pdf", ".md", ".docx"}


def load_docx(path: Path) -> List[Document]:
    document = docxDocument(path)
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    return [Document(page_content=text, metadata={"source": str(path), "title": path.name})]

    
def load_file(path: Path) -> List[Document]:
    extension = path.suffix.lower()

    if extension == ".pdf":
        docs = PyPDFLoader(str(path)).load()

    elif extension in [".txt", ".md"]:
        docs = TextLoader(str(path), encoding="utf-8").load()

    elif extension == ".docx":
        docs = load_docx(path)

    else:
        raise ValueError(f"File extension {extension} not in valid file extensions:  PDF, TXT, MD, or DOCX.")

    for d in docs:
        d.metadata.setdefault("source", str(path))
        d.metadata.setdefault("title", path.name)
        d.metadata["document_name"] = path.name

    return docs


def generate_stable_ids(path: Path, chunk: Document, position: int) -> str:

    material = f"{path.name}|{position}|{chunk.page_content}".encode("utf-8")
    digest = sha256(material).hexdigest()[:24]
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "-" for c in path.stem)[:60]

    return f"{safe_stem}--{position}--{digest}"



def ingest_file(path: Path) -> int:

    s = get_settings()
    docs = load_file(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 160,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(docs)
    if not chunks:
        return 0

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk index"] = i
        chunk.metadata["knowledge_base"] = s.PINECONE_NAMESPACE

    ids = [generate_stable_ids(path, chunk, i) for i, chunk in enumerate(chunks)]

    get_vector_store().add_documents(chunks, ids = ids)

    return len(chunks)


def ingest_dir(dir: Path) -> int:
    total = 0

    if not dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {dir}")
    if not dir.is_dir():
        raise NotADirectoryError(f"Document path is not a directory: {dir}")

    for path in sorted(dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            total += ingest_file(path)

    return total

def namespace() -> str:
    return get_settings().PINECONE_NAMESPACE