from pydantic import BaseModel
from functools import lru_cache
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL_ID: str = os.getenv("GEMINI_MODEL_ID", "Gemini 3.8 Flash")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL" , "gemini-embedding-2") 
    EMBEDDING_DIM : int = os.getenv("EMBEDDING_DIM", 3072)


    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "cloud-ops-self-rag")
    PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "")
    PINECONE_CLOUD: str =os.getenv("PINECONE_CLOUD","aws") 
    PINECONE_REGION: str =os.getenv("PINECONE_REGION","us-east-1") 

    TAVILY_API_KEY: str =os.getenv("TAVILY_API_KEY", "") 

    TOP_K: int =os.getenv("TOP_K", "5") 
    MAX_SUPPORT_RETRIES: int =os.getenv("MAX_SUPPORT_RETRIES", "2") 
    MAX_RETRIVAL_RETRIES: int =os.getenv("MAX_RETRIVAL_RETRIES", "2") 
    MAX_WEB_RETRIES: int =os.getenv("MAX_WEB_RETRIES", "2") 
    DB_PATH: str =os.getenv("DB_PATH", "data/audit.db") 

    @property
    def db_file(self) -> Path:
        p = Path(self.DB_PATH)
        return p if p.is_absolute() else ROOT/p


@lru_cache
def get_settings() -> Settings:
    return Settings()