from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from src.config import get_settings
from langchain_pinecone import PineconeVectorStore
from functools import lru_cache


@lru_cache
def get_embeddings():
    s = get_settings()

    if not s.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing")

    return GoogleGenerativeAIEmbeddings(
        model=s.GEMINI_EMBEDDING_MODEL,
        api_key=s.GEMINI_API_KEY,
        output_dimensionality=s.EMBEDDING_DIM
    )


def get_pinecone_client():
    s = get_settings()
    if not s.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is missing")

    return Pinecone(api_key=s.PINECONE_API_KEY)

def ensure_pinecone_index():
    s = get_settings()
    pc = get_pinecone_client()
    existing = {x.name for x in pc.list_indexes()}



    # print(s.EMBEDDING_DIM)
    # print(type(s.EMBEDDING_DIM))
    if s.PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=s.PINECONE_INDEX_NAME,
            dimension=int(s.EMBEDDING_DIM),
            metric="cosine",
            spec=ServerlessSpec(cloud=s.PINECONE_CLOUD, region=s.PINECONE_REGION)
        )

    else:
        desc = pc.describe_index(s.PINECONE_INDEX_NAME)
        existing_dim = getattr(desc, "dimension", None)

        if existing_dim is None and isinstance(desc, dict):
            existing_dim = desc.get("dimension")

        if existing_dim and int(existing_dim) != int(s.EMBEDDING_DIM):
            raise RuntimeError(
                f"pinecone Vector index: {s.PINECONE_INDEX_NAME} has dim : {existing_dim},"
                f'but {s.GEMINI_EMBEDDING_MODEL} is configured for {s.EMBEDDING_DIM},'
                "use a new index name or recreate index with correct dimension"
            )

    return pc.Index(s.PINECONE_INDEX_NAME)


def get_vector_store():
    s = get_settings()

    index = ensure_pinecone_index()

    return PineconeVectorStore(
        index=index,
        embedding=get_embeddings(),
        namespace= s.PINECONE_NAMESPACE
    )


def get_retriever():
    s = get_settings()

    return get_vector_store().as_retriever(search_kwargs = {"k": s.TOP_K})