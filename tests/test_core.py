import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from docx import Document as WordDocument
from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_google_genai.chat_models import GoogleAPIError
from pydantic import ValidationError

from src.app import chat
from src.config import Settings
from src.db import init_db, latest_audits, save_audit
from src.ingestion import ingest_dir, ingest_file, load_docx, namespace
from src.models import ChatRequest
from src.self_rag import RelevanceDecision, _llm, decide_retrival, grade_relevance
from src.vectorstore import get_embeddings


class ConfigurationTests(unittest.TestCase):
    def test_numeric_settings_are_integers(self):
        settings = Settings()

        self.assertIsInstance(settings.EMBEDDING_DIM, int)
        self.assertIsInstance(settings.GEMINI_TIMEOUT_SECONDS, int)
        self.assertIsInstance(settings.GEMINI_MAX_RETRIES, int)
        self.assertIsInstance(settings.TOP_K, int)
        self.assertIsInstance(settings.MAX_SUPPORT_RETRIES, int)
        self.assertIsInstance(settings.MAX_RETRIVAL_RETRIES, int)
        self.assertIsInstance(settings.MAX_WEB_RETRIES, int)

    def test_llm_uses_configured_timeout(self):
        settings = SimpleNamespace(
            GEMINI_API_KEY="test-key",
            GEMINI_MODEL_ID="test-model",
            GEMINI_TIMEOUT_SECONDS=17,
            GEMINI_MAX_RETRIES=1,
        )
        with patch("src.self_rag.get_settings", return_value=settings):
            with patch("src.self_rag.ChatGoogleGenerativeAI") as llm:
                _llm()

        self.assertEqual(llm.call_args.kwargs["timeout"], 17)
        self.assertEqual(llm.call_args.kwargs["max_retries"], 1)

    def test_embeddings_use_configured_request_timeout(self):
        settings = SimpleNamespace(
            GEMINI_API_KEY="test-key",
            GEMINI_EMBEDDING_MODEL="test-embedding-model",
            EMBEDDING_DIM=64,
            GEMINI_TIMEOUT_SECONDS=17,
        )
        get_embeddings.cache_clear()
        self.addCleanup(get_embeddings.cache_clear)
        with patch("src.vectorstore.get_settings", return_value=settings):
            with patch("src.vectorstore.GoogleGenerativeAIEmbeddings") as embeddings:
                get_embeddings()

        self.assertEqual(
            embeddings.call_args.kwargs["request_options"],
            {"timeout": 17_000},
        )

    def test_relative_database_path_uses_current_directory(self):
        settings = Settings(DB_PATH="data/audit.db")
        self.assertEqual(settings.db_file, Path.cwd() / "data" / "audit.db")


class RequestValidationTests(unittest.TestCase):
    def test_chat_request_strips_whitespace_before_length_validation(self):
        request = ChatRequest(question="  check latency  ", thread_id="  thread-1  ")
        self.assertEqual(request.question, "check latency")
        self.assertEqual(request.thread_id, "thread-1")

        with self.assertRaises(ValidationError):
            ChatRequest(question="  ", thread_id="thread-1")


class ChatEndpointTests(unittest.TestCase):
    def test_gemini_timeout_returns_gateway_timeout(self):
        async def invoke_chat():
            with patch(
                "src.app.run_in_threadpool",
                new=AsyncMock(
                    side_effect=GoogleAPIError(
                        504,
                        {"message": "Deadline expired before operation could complete."},
                    )
                ),
            ):
                with self.assertRaises(HTTPException) as raised:
                    await chat(ChatRequest(
                        question="How do I troubleshoot an incident?",
                        thread_id="test-thread",
                    ))
            return raised.exception

        error = asyncio.run(invoke_chat())
        self.assertEqual(error.status_code, 504)
        self.assertIn("Gemini timed out", error.detail)


class IngestionTests(unittest.TestCase):
    def test_docx_loader_extracts_paragraph_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "runbook.docx"
            document = WordDocument()
            document.add_paragraph("First step")
            document.add_paragraph("Second step")
            document.save(path)

            documents = load_docx(path)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "First step\nSecond step")

    def test_empty_file_does_not_initialize_vector_store(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.txt"
            path.touch()
            with patch("src.ingestion.get_vector_store") as get_vector_store:
                self.assertEqual(ingest_file(path), 0)

        get_vector_store.assert_not_called()

    def test_ingest_dir_uses_supported_extensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "runbook.md").touch()
            (directory / "ignore.csv").touch()
            with patch("src.ingestion.ingest_file", return_value=3) as ingest:
                self.assertEqual(ingest_dir(directory), 3)

        ingest.assert_called_once_with(directory / "runbook.md")

    def test_ingest_dir_reports_missing_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing"
            with self.assertRaises(FileNotFoundError):
                ingest_dir(missing)

    def test_namespace_uses_configured_field(self):
        with patch(
            "src.ingestion.get_settings",
            return_value=SimpleNamespace(PINECONE_NAMESPACE="runbooks"),
        ):
            self.assertEqual(namespace(), "runbooks")


class SelfRagTests(unittest.TestCase):
    def test_retrieval_decision_timeout_continues_with_evidence_retrieval(self):
        grader = Mock()
        grader.invoke.side_effect = GoogleAPIError(
            504,
            {"message": "Deadline expired before operation could complete."},
        )
        llm = Mock()
        llm.with_structured_output.return_value = grader

        with patch("src.self_rag._llm", return_value=llm):
            result = decide_retrival({
                "question": "How do I troubleshoot a production incident?",
                "trace": [],
            })

        self.assertTrue(result["need_retrieval"])
        self.assertIn("continuing with evidence retrieval", result["trace"][-1])

    def test_retrieval_decision_reraises_non_timeout_provider_errors(self):
        grader = Mock()
        grader.invoke.side_effect = GoogleAPIError(503, {"message": "Unavailable"})
        llm = Mock()
        llm.with_structured_output.return_value = grader

        with patch("src.self_rag._llm", return_value=llm):
            with self.assertRaises(GoogleAPIError):
                decide_retrival({
                    "question": "How do I troubleshoot a production incident?",
                    "trace": [],
                })

    def test_relevance_grades_all_documents_in_one_call(self):
        documents = [
            Document(page_content="Unrelated note"),
            Document(page_content="Useful runbook detail"),
        ]
        grader = Mock()
        grader.invoke.return_value = SimpleNamespace(relevant_indices=[1, 99])
        llm = Mock()
        llm.with_structured_output.return_value = grader

        with patch("src.self_rag._llm", return_value=llm):
            result = grade_relevance({
                "question": "How do I respond?",
                "docs": documents,
                "trace": [],
                "source_mode": "internal",
            })

        llm.with_structured_output.assert_called_once_with(RelevanceDecision)
        grader.invoke.assert_called_once()
        self.assertEqual(result["relevant_docs"], [documents[1]])


class AuditDatabaseTests(unittest.TestCase):
    def test_audits_use_configured_database_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = SimpleNamespace(db_file=Path(temp_dir) / "nested" / "audit.db")
            with patch("src.db.get_settings", return_value=settings):
                init_db()
                save_audit("test question", {"answer": "test answer"})
                audits = latest_audits(limit=1)

        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0]["question"], "test question")
        self.assertEqual(audits[0]["answer"], "test answer")


if __name__ == "__main__":
    unittest.main()
