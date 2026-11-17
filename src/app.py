import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_google_genai.chat_models import GoogleAPIError

from src.db import init_db, latest_audits, save_audit
from src.ingestion import SUPPORTED, ingest_file, namespace
from src.models import ChatRequest, ChatResponse, UploadResponse
from src.self_rag import run_self_rag

logger = logging.getLogger(__name__)

load_dotenv()

PACKAGE_ROOT = Path(__file__).resolve().parent
UPLOADS = Path(os.getenv("UPLOAD_DIR", str(Path.cwd() / "uploads")))
UPLOADS.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="CloudOps Sentinel — Enterprise Incident Response Self-RAG Copilot",
    version="2.0.0",
    description="Self-RAG copilot for cloud operations, production troubleshooting, and incident-response runbooks.",
    lifespan=lifespan,
)
app.mount(
    "/static",
    StaticFiles(directory=str(PACKAGE_ROOT / "static")),
    name="static",
)
templates = Jinja2Templates(directory=str(PACKAGE_ROOT / "templates"))


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "cloudops-sentinel-self-rag"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    started_at = perf_counter()
    logger.info("Chat request started")
    try:
        result = await run_in_threadpool(
            run_self_rag,
            payload.question,
            payload.thread_id,
        )
        await run_in_threadpool(save_audit, payload.question, result)
        logger.info("Chat request completed in %.1f seconds", perf_counter() - started_at)
        return ChatResponse(**result)
    except GoogleAPIError as error:
        logger.exception("Gemini request failed after %.1f seconds", perf_counter() - started_at)
        if error.code == 504:
            raise HTTPException(
                status_code=504,
                detail="Gemini timed out while generating a response. Please try again.",
            ) from error
        raise HTTPException(
            status_code=502,
            detail="Gemini could not complete the request. Please try again.",
        ) from error
    except Exception as error:
        logger.exception("Chat request failed after %.1f seconds", perf_counter() - started_at)
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if not filename or filename in {".", ".."} or ":" in filename:
        raise HTTPException(status_code=400, detail="A valid filename is required")

    if Path(filename).suffix.lower() not in SUPPORTED:
        raise HTTPException(status_code=400, detail="Supported: PDF, TXT, MD, DOCX")

    target = UPLOADS / filename
    with target.open("wb") as output:
        shutil.copyfileobj(file.file, output)

    try:
        count = await run_in_threadpool(ingest_file, target)
        return UploadResponse(
            filename=filename,
            chunks_indexed=count,
            namespace=namespace(),
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/api/audits")
def audits(limit: int = 20):
    return latest_audits(min(max(limit, 1), 100))


def main() -> None:
    import uvicorn

    uvicorn.run(
        "src.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
    )
