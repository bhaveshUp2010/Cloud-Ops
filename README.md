## CloudOps Sentinel

CloudOps Sentinel is a FastAPI incident-response assistant that retrieves
operations guidance from a Pinecone knowledge base, optionally searches the
web with Tavily, and stores chat audits and graph memory in SQLite.

### Run locally

Use Python 3.11 or newer from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app:app --reload
```

Set the provider credentials in `.env` before using chat or document ingestion.
Gemini and Pinecone are required for retrieval; Tavily is optional. The health
endpoint and UI can run without provider credentials.

The app listens on `http://127.0.0.1:8000` with the command above. To install
the project and use its console command, run `uv sync` and then `uv run cloud-ops`.

### Documents and storage

Upload PDF, TXT, Markdown, or DOCX files through the UI. The default local
database is `data/audit.db`, and uploads are stored in `uploads/`. Set
`DB_PATH` or `UPLOAD_DIR` to change those locations.

To index all supported files under `documents/` from the command line:

```powershell
python data_ingestion.py
```

The app exposes `/api/health`, `/api/chat`, `/api/upload`, and `/api/audits`.