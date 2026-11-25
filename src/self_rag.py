
import logging
from typing import List, Annotated, Literal, TypedDict
import operator
import sqlite3

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleAPIError
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from tavily import TavilyClient

from src.config import get_settings
from src.vectorstore import get_retriever

logger = logging.getLogger(__name__)


# ============================================================
# STATE
# ============================================================

class RagState(TypedDict, total=False):
    user_question: str
    question: str

    memory: Annotated[List[str], operator.add]

    retrieval_query: str
    web_query: str
    need_retrieval: bool

    docs: List[Document]
    relevant_docs: List[Document]

    context: str
    answer: str

    support_status: Literal[
        "fully_supported",
        "partially_supported",
        "no_support",
        ""
    ]

    evidence: List[str]

    usefulness: Literal["useful", "not_useful", ""]

    use_reason: str

    support_retries: int
    retrieval_rewrites: int
    web_rewrites: int

    source_mode: Literal["internal", "web", "direct", "none"]

    used_web_search: bool
    trace: List[str]


# ============================================================
# PYDANTIC STRUCTURED OUTPUT SCHEMAS
# ============================================================

class RetrieveDescision(BaseModel):
    should_retrieve: bool


class RelevanceDecision(BaseModel):
    relevant_indices: List[int] = Field(default_factory=list)


class SupportDescison(BaseModel):
    status: Literal[
        "fully_supported",
        "partially_supported",
        "no_support"
    ]

    evidence: List[str] = Field(default_factory=list)


class UsefullnessDescision(BaseModel):
    status: Literal["useful", "not_useful"]
    reason: str


class QueryRewerite(BaseModel):
    query: str


# ============================================================
# LLM
# ============================================================

def _llm():
    s = get_settings()

    if not s.GEMINI_API_KEY:
        raise RuntimeError("GEMINI API KEY IS MISSING")

    return ChatGoogleGenerativeAI(
        model=s.GEMINI_MODEL_ID,
        google_api_key=s.GEMINI_API_KEY,
        temperature=0,
        timeout=s.GEMINI_TIMEOUT_SECONDS,
        max_retries=s.GEMINI_MAX_RETRIES,
    )


# ============================================================
# HELPERS
# ============================================================

def _trace(state: RagState, item: str):
    logger.info("Self-RAG progress: %s", item.partition(":")[0])
    return [*(state.get("trace") or []), item]


def _format_context(docs: List[Document]) -> str:
    blocks = []

    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}

        if meta.get("source_type") == "web":
            head = (
                f"[WEB {i}] "
                f"{meta.get('title', '')} | "
                f"{meta.get('url', '')}"
            )

        else:
            head = (
                f"[INTERNAL {i}] "
                f"{meta.get('title') or meta.get('document_name') or meta.get('source', '')}"
            )

            if meta.get("page") is not None:
                head += f" | page {int(meta['page']) + 1}"

        blocks.append(f"{head}\n{d.page_content}")

    return "\n\n---\n\n".join(blocks)


def _memory_text(state: RagState, limit: int = 5) -> str:
    items = state.get("memory") or []

    return (
        "\n\n".join(items[-limit:])
        if items
        else "No previous conversation context."
    )


def _extract_text(content) -> str:
    """
    Convert LangChain/Gemini message content into a plain string.

    Gemini may return either a normal string or a list of content
    blocks such as [{"type": "text", "text": "..."}].
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        text_value = content.get("text")
        return text_value if isinstance(text_value, str) else ""

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text_value = block.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)

        return "\\n".join(part for part in parts if part).strip()

    return str(content)


# ============================================================
# NODE 1: CONTEXTUALIZE QUESTION
# ============================================================

def contextualize_question(state: RagState):

    history = _memory_text(state)

    user_q = (
        state.get("user_question")
        or state.get("question", "")
    )

    if not state.get("memory"):

        return {
            "question": user_q,
            "trace": _trace(
                state,
                "Memory: new incident session"
            )
        }

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rewrite the newest user message as a standalone "
            "cloud-operations question using previous conversation "
            "only when needed. Preserve service names, symptoms, "
            "errors, and constraints. If the message already stands "
            "alone, return it unchanged. Do not answer."
        ),
        (
            "human",
            "Previous conversation:\n{history}\n\n"
            "Newest message:\n{question}"
        ),
    ])

    out = _llm().with_structured_output(
        QueryRewerite
    ).invoke(
        prompt.format_messages(
            history=history,
            question=user_q
        )
    )

    return {
        "question": out.query,
        "trace": _trace(
            state,
            f"Memory contextualized question: {out.query}"
        )
    }


# ============================================================
# NODE 2: COMMIT MEMORY
# ============================================================

def commit_memory(state: RagState):

    user_q = (
        state.get("user_question")
        or state.get("question", "")
    )

    answer = state.get("answer", "")
    route = state.get("source_mode", "none")

    entry = (
        f"User: {user_q}\n"
        f"Assistant ({route}): {answer}"
    )

    return {
        "memory": [entry],
        "trace": _trace(
            state,
            "SQLite memory checkpoint updated"
        )
    }


# ============================================================
# NODE 3: DECIDE RETRIEVAL
# ============================================================

def decide_retrival(state: RagState):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Decide whether the question needs retrieval. "
            "Choose true for cloud operations, production incidents, "
            "service runbooks, deployment procedures, infrastructure "
            "behavior, troubleshooting, specific/current technical "
            "facts, or whenever evidence is needed. "
            "Choose false only for generic technical explanations "
            "that can safely be answered from general knowledge. "
            "If unsure choose true."
        ),
        ("human", "Question: {question}"),
    ])

    try:
        out = _llm().with_structured_output(
            RetrieveDescision
        ).invoke(
            prompt.format_messages(
                question=state["question"]
            )
        )
    except GoogleAPIError as error:
        if error.code != 504:
            raise

        logger.warning(
            "Gemini retrieval decision timed out; continuing with evidence retrieval"
        )
        return {
            "need_retrieval": True,
            "trace": _trace(
                state,
                "Retrieval decision unavailable; continuing with evidence retrieval"
            )
        }

    return {
        "need_retrieval": out.should_retrieve,
        "trace": _trace(
            state,
            f"Retrieval decision: {out.should_retrieve}"
        )
    }


def route_after_decide(
    state: RagState
) -> Literal["retrieve", "direct"]:

    return (
        "retrieve"
        if state.get("need_retrieval", True)
        else "direct"
    )


# ============================================================
# NODE 4: DIRECT ANSWER
# ============================================================

def generate_direct(state: RagState):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer briefly using general technical knowledge only. "
            "Do not invent organization-specific infrastructure, "
            "runbooks, credentials, incident history, or procedures."
        ),
        ("human", "{question}"),
    ])

    response = _llm().invoke(
        prompt.format_messages(
            question=state["question"]
        )
    )
    ans = _extract_text(response.content)

    return {
        "answer": ans,
        "source_mode": "direct",
        "support_status": "",
        "usefulness": "",
        "trace": _trace(
            state,
            "Generated direct answer"
        )
    }


# ============================================================
# NODE 5: INTERNAL RETRIEVAL
# ============================================================

def retrieve_internal(state: RagState):

    query = (
        state.get("retrieval_query")
        or state["question"]
    )

    docs = get_retriever().invoke(query)

    for d in docs:
        d.metadata = {
            **(d.metadata or {}),
            "source_type": "internal"
        }

    return {
        "docs": docs,
        "relevant_docs": [],
        "source_mode": "internal",
        "trace": _trace(
            state,
            f"Internal retrieval: {len(docs)} chunks"
        )
    }


# ============================================================
# NODE 6: GRADE RELEVANCE
# ============================================================

def grade_relevance(state: RagState):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Select the zero-based indices of documents that contain "
            "information useful for answering the question. Do not "
            "require the exact final answer. Be strict about unrelated "
            "content. Return an empty list when no document is relevant."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Documents:\n{documents}"
        ),
    ])

    grader = _llm().with_structured_output(
        RelevanceDecision
    )

    docs = state.get("docs", [])
    if docs:
        formatted_docs = "\n\n".join(
            f"[{index}]\n{doc.page_content[:7000]}"
            for index, doc in enumerate(docs)
        )
        decision = grader.invoke(
            prompt.format_messages(
                question=state["question"],
                documents=formatted_docs,
            )
        )
        relevant_indices = {
            index
            for index in decision.relevant_indices
            if 0 <= index < len(docs)
        }
        relevant_docs = [
            doc for index, doc in enumerate(docs)
            if index in relevant_indices
        ]
    else:
        relevant_docs = []

    mode = state.get("source_mode", "internal")

    return {
        "relevant_docs": relevant_docs,
        "trace": _trace(
            state,
            f"Relevance grade ({mode}): "
            f"{len(relevant_docs)}/{len(state.get('docs', []))} relevant"
        )
    }


# ============================================================
# NODE 7: ROUTE AFTER RELEVANCE
# ============================================================

def route_after_relevance(
    state: RagState
) -> Literal[
    "generate",
    "rewrite_web",
    "rewrite_internal",
    "no_answer"
]:

    if state.get("relevant_docs"):
        return "generate"

    s = get_settings()

    if state.get("source_mode") == "web":

        if state.get("web_rewrites", 0) < s.MAX_WEB_RETRIES:
            return "rewrite_web"

        return "no_answer"

    if state.get("retrieval_rewrites", 0) < s.MAX_RETRIVAL_RETRIES:
        return "rewrite_internal"

    return "rewrite_web"


# ============================================================
# NODE 8: REWRITE INTERNAL QUERY
# ============================================================

def rewrite_internal_query(state: RagState):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rewrite the operations question for semantic vector "
            "retrieval over internal cloud runbooks, SOPs, "
            "postmortems, architecture notes, and troubleshooting "
            "documents. Use 6-18 words, preserve service names "
            "and error symptoms, add useful operations keywords, "
            "remove filler, and do not answer."
        ),
        (
            "human",
            "Question: {question}\n"
            "Previous query: {previous}"
        ),
    ])

    out = _llm().with_structured_output(
        QueryRewerite
    ).invoke(
        prompt.format_messages(
            question=state["question"],
            previous=state.get("retrieval_query", "")
        )
    )

    return {
        "retrieval_query": out.query,
        "retrieval_rewrites": (
            state.get("retrieval_rewrites", 0) + 1
        ),
        "docs": [],
        "relevant_docs": [],
        "trace": _trace(
            state,
            f"Rewrote internal query: {out.query}"
        )
    }


# ============================================================
# NODE 9: REWRITE WEB QUERY
# ============================================================

def rewrite_web_query(state: RagState):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rewrite the question into a concise internet search "
            "query of 6-14 words. Preserve important entities. "
            "Add recency wording when the question asks for "
            "latest/current/today. Do not answer."
        ),
        (
            "human",
            "Question: {question}\n"
            "Previous web query: {previous}"
        ),
    ])

    out = _llm().with_structured_output(
        QueryRewerite
    ).invoke(
        prompt.format_messages(
            question=state["question"],
            previous=state.get("web_query", "")
        )
    )

    return {
        "web_query": out.query,
        "web_rewrites": (
            state.get("web_rewrites", 0) + 1
        ),
        "docs": [],
        "relevant_docs": [],
        "trace": _trace(
            state,
            f"Prepared internet search query: {out.query}"
        )
    }


# ============================================================
# NODE 10: WEB SEARCH
# ============================================================

def web_search(state: RagState):

    s = get_settings()

    if not s.TAVILY_API_KEY:

        return {
            "docs": [],
            "source_mode": "web",
            "used_web_search": False,
            "trace": _trace(
                state,
                "Internet search unavailable: TAVILY_API_KEY missing"
            )
        }

    client = TavilyClient(
        api_key=s.TAVILY_API_KEY
    )

    query = (
        state.get("web_query")
        or state["question"]
    )

    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=5,
        include_answer=False
    )

    docs = []

    for result in response.get("results", []):

        docs.append(
            Document(
                page_content=result.get("content", ""),
                metadata={
                    "source_type": "web",
                    "source": result.get("url", ""),
                    "url": result.get("url", ""),
                    "title": result.get("title", "")
                }
            )
        )

    return {
        "docs": docs,
        "source_mode": "web",
        "used_web_search": True,
        "trace": _trace(
            state,
            f"Internet search: {len(docs)} results"
        )
    }


# ============================================================
# NODE 11: GENERATE FROM CONTEXT
# ============================================================

def generate_from_context(state: RagState):

    context = _format_context(
        state.get("relevant_docs", [])
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are CloudOps Sentinel, an enterprise cloud "
            "operations and incident-response copilot. "
            "Answer using only the supplied evidence. "
            "Prefer private runbooks, SOPs, architecture notes, "
            "and postmortems when present. "
            "If evidence comes from the web, clearly label it "
            "as external guidance and never present it as an "
            "organization-specific procedure. "
            "Do not invent infrastructure facts, credentials, "
            "commands, or incident history. "
            "Provide concise, actionable troubleshooting guidance "
            "and preserve cautions contained in the evidence."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Evidence:\n{context}"
        ),
    ])

    response = _llm().invoke(
        prompt.format_messages(
            question=state["question"],
            context=context
        )
    )
    ans = _extract_text(response.content)

    return {
        "answer": ans,
        "context": context,
        "support_retries": 0,
        "trace": _trace(
            state,
            f"Generated answer from {state.get('source_mode', '')} evidence"
        )
    }


# ============================================================
# NODE 12: CHECK SUPPORT
# ============================================================

def check_support(state: RagState):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Verify whether every meaningful claim in the answer "
            "is supported by the supplied evidence. "
            "Return fully_supported only when all important claims "
            "are grounded; partially_supported when some claims "
            "are grounded but others are not; no_support when "
            "key claims are unsupported. Evidence excerpts should "
            "be short."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Answer:\n{answer}\n\n"
            "Evidence:\n{context}"
        ),
    ])

    out = _llm().with_structured_output(
        SupportDescison
    ).invoke(
        prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", "")
        )
    )

    return {
        "support_status": out.status,
        "evidence": out.evidence,
        "trace": _trace(
            state,
            f"Support check: {out.status}"
        )
    }


def route_after_support(
    state: RagState
) -> Literal["usefulness", "revise"]:

    if state.get("support_status") == "fully_supported":
        return "usefulness"

    if (
        state.get("support_retries", 0)
        >= get_settings().MAX_SUPPORT_RETRIES
    ):
        return "usefulness"

    return "revise"


# ============================================================
# NODE 13: REVISE ANSWER
# ============================================================

def revise_answer(state: RagState):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rewrite the answer so every factual claim is directly "
            "supported by the provided evidence. Remove unsupported "
            "interpretation and speculation. Still answer naturally. "
            "Do not mention the verification process."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Current answer:\n{answer}\n\n"
            "Evidence:\n{context}"
        ),
    ])

    response = _llm().invoke(
        prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", "")
        )
    )
    ans = _extract_text(response.content)

    return {
        "answer": ans,
        "support_retries": (
            state.get("support_retries", 0) + 1
        ),
        "trace": _trace(
            state,
            "Revised answer for grounding"
        )
    }


# ============================================================
# NODE 14: CHECK USEFULNESS
# ============================================================

def check_usefulness(state: RagState):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Judge only whether the answer directly addresses "
            "the user's question. Do not re-grade factual grounding. "
            "Return useful or not_useful and a one-line reason."
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Answer:\n{answer}"
        ),
    ])

    out = _llm().with_structured_output(
        UsefullnessDescision
    ).invoke(
        prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", "")
        )
    )

    return {
        "usefulness": out.status,
        "use_reason": out.reason,
        "trace": _trace(
            state,
            f"Usefulness check: {out.status}"
        )
    }


def route_after_usefulness(
    state: RagState
) -> Literal[
    "end",
    "rewrite_internal",
    "rewrite_web",
    "no_answer"
]:

    if state.get("usefulness") == "useful":
        return "end"

    s = get_settings()

    if state.get("source_mode") == "internal":

        if state.get("retrieval_rewrites", 0) < s.MAX_RETRIVAL_RETRIES:
            return "rewrite_internal"

        return "rewrite_web"

    if (
        state.get("source_mode") == "web"
        and state.get("web_rewrites", 0) < s.MAX_WEB_RETRIES
    ):
        return "rewrite_web"

    return "no_answer"


# ============================================================
# NODE 15: NO ANSWER
# ============================================================

def no_answer(state: RagState):

    return {
        "answer": (
            "I could not find enough reliable runbook or "
            "external evidence to recommend a safe "
            "troubleshooting action."
        ),
        "source_mode": "none",
        "trace": _trace(
            state,
            "Stopped: no reliable answer found"
        )
    }


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph():

    g = StateGraph(RagState)

    # Register nodes
    g.add_node("contextualize", contextualize_question)
    g.add_node("decide_retrival", decide_retrival)
    g.add_node("direct", generate_direct)
    g.add_node("retrieve", retrieve_internal)
    g.add_node("grade", grade_relevance)
    g.add_node("rewrite_internal", rewrite_internal_query)
    g.add_node("rewrite_web", rewrite_web_query)
    g.add_node("web_search", web_search)
    g.add_node("generate", generate_from_context)
    g.add_node("support", check_support)
    g.add_node("revise", revise_answer)
    g.add_node("usefulness", check_usefulness)
    g.add_node("no_answer", no_answer)
    g.add_node("commit_memory", commit_memory)

    # Entry
    g.add_edge(START, "contextualize")

    # Retrieval decision
    g.add_edge("contextualize", "decide_retrival")

    g.add_conditional_edges(
        "decide_retrival",
        route_after_decide,
        {
            "direct": "direct",
            "retrieve": "retrieve"
        }
    )

    # Direct answer
    g.add_edge("direct", "commit_memory")

    # Internal retrieval
    g.add_edge("retrieve", "grade")

    g.add_conditional_edges(
        "grade",
        route_after_relevance,
        {
            "generate": "generate",
            "rewrite_internal": "rewrite_internal",
            "rewrite_web": "rewrite_web",
            "no_answer": "no_answer"
        }
    )

    # Query rewriting
    g.add_edge("rewrite_internal", "retrieve")
    g.add_edge("rewrite_web", "web_search")

    # Web retrieval
    g.add_edge("web_search", "grade")

    # Generation and verification
    g.add_edge("generate", "support")

    g.add_conditional_edges(
        "support",
        route_after_support,
        {
            "usefulness": "usefulness",
            "revise": "revise"
        }
    )

    g.add_edge("revise", "support")

    # Usefulness routing
    g.add_conditional_edges(
        "usefulness",
        route_after_usefulness,
        {
            "end": "commit_memory",
            "rewrite_internal": "rewrite_internal",
            "rewrite_web": "rewrite_web",
            "no_answer": "no_answer"
        }
    )

    # Finalization
    g.add_edge("no_answer", "commit_memory")
    g.add_edge("commit_memory", END)

    # SQLite checkpointing
    db_path = get_settings().db_file.parent / "langgraph_memory.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path),
        check_same_thread=False
    )

    checkpointer = SqliteSaver(conn)

    return g.compile(
        checkpointer=checkpointer
    )


# ============================================================
# SOURCE EXTRACTION
# ============================================================

def _sources(docs: List[Document]):

    seen = set()
    out = []

    for d in docs or []:

        meta = d.metadata or {}

        typ = (
            "web"
            if meta.get("source_type") == "web"
            else "internal"
        )

        key = (
            typ,
            meta.get("url") or meta.get("source"),
            meta.get("page")
        )

        if key in seen:
            continue

        seen.add(key)

        item = {
            "type": typ,
            "title": (
                meta.get("title")
                or meta.get("document_name")
                or ""
            ),
            "source": meta.get("source") or "",
            "url": (
                meta.get("url")
                if typ == "web"
                else None
            ),
            "page": (
                int(meta["page"]) + 1
                if typ == "internal"
                and meta.get("page") is not None
                else None
            )
        }

        out.append(item)

    return out


# ============================================================
# GRAPH INSTANCE
# ============================================================

_graph = None


# ============================================================
# RUN SELF RAG
# ============================================================

def run_self_rag(question: str, thread_id: str) -> dict:

    global _graph

    if _graph is None:
        _graph = build_graph()

    initial: RagState = {
        "user_question": question,
        "question": question,

        "memory": [],

        "retrieval_query": question,
        "web_query": "",

        "need_retrieval": True,

        "docs": [],
        "relevant_docs": [],

        "context": "",
        "answer": "",

        "support_status": "",
        "evidence": [],

        "usefulness": "",
        "use_reason": "",

        "support_retries": 0,
        "retrieval_rewrites": 0,
        "web_rewrites": 0,

        "source_mode": "internal",
        "used_web_search": False,

        "trace": []
    }

    result = _graph.invoke(
        initial,
        config={
            "configurable": {
                "thread_id": thread_id
            },
            "recursion_limit": 60
        }
    )

    mode = result.get("source_mode", "none")

    route = {
        "internal": "Private Runbooks",
        "web": "Internet Search",
        "direct": "General Knowledge",
        "none": "No Reliable Evidence"
    }.get(mode, mode)

    return {
        "answer": result.get("answer", ""),
        "route": route,
        "used_web_search": bool(
            result.get("used_web_search")
        ),
        "support_status": result.get(
            "support_status", ""
        ),
        "usefulness": result.get(
            "usefulness", ""
        ),
        "sources": _sources(
            result.get("relevant_docs", [])
        ),
        "trace": result.get("trace", []),
        "thread_id": thread_id,
        "memory_turns": len(
            result.get("memory", [])
        )
    }
