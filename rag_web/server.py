"""Web API and page for the Pinecone RAG.

    python rag_web/server.py
    open http://localhost:8000

The RAG itself lives in embeddings_app/rag_pinecone.py - this only exposes it
over HTTP and serves the page. Question answering goes through
rag_pinecone.ask(), the same function the CLI uses, so the two can't diverge.

Endpoints:
    GET  /            the page
    GET  /api/status  index name, vector count, whether it's ready
    POST /api/ask     {question, k} -> {answer, sources[]}
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The RAG code lives one folder over; add it to the path rather than copying
# any of it here.
sys.path.insert(0, str(ROOT / "embeddings_app"))

import core                                    # noqa: E402  (loads .env)
import rag_pinecone as rag                     # noqa: E402

from fastapi import FastAPI                    # noqa: E402
from fastapi.responses import FileResponse     # noqa: E402
from fastapi.staticfiles import StaticFiles    # noqa: E402
from pydantic import BaseModel, Field          # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="RAG over docs/")

# Opening the store creates an embeddings client and a Pinecone connection, so
# it is done once and reused rather than per request. Kept lazy so the server
# still starts (and can report the problem) when Pinecone is unreachable.
_store = None
_error: str | None = None


def store():
    global _store, _error
    if _store is None and _error is None:
        try:
            _store = rag.open_store(rag.INDEX_NAME, "")
        except SystemExit as exc:      # open_store raises this when unbuilt
            _error = str(exc)
        except Exception as exc:
            _error = f"{type(exc).__name__}: {exc}"
    return _store


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)


@app.get("/api/status")
def status() -> dict:
    """What the page shows in its header, and whether asking will work."""
    try:
        pc = rag.client()
        if not pc.has_index(rag.INDEX_NAME):
            return {"ready": False,
                    "error": f"Index '{rag.INDEX_NAME}' does not exist. "
                             f"Run: python embeddings_app/rag_pinecone.py --build"}
        stats = pc.Index(rag.INDEX_NAME).describe_index_stats()
        return {
            "ready": True,
            "index": rag.INDEX_NAME,
            "vectors": stats.get("total_vector_count", 0),
            "model": core.CHAT_MODEL,
            "embed_model": core.EMBED_MODEL,
            "chunking": "one chunk per paragraph",
        }
    except SystemExit as exc:
        return {"ready": False, "error": str(exc)}
    except Exception as exc:
        return {"ready": False, "error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/ask")
def ask(q: Question) -> dict:
    s = store()
    if s is None:
        return {"error": _error or "The index is not available."}
    try:
        # The CLI's own function - no retrieval or prompting logic here.
        return rag.ask(s, q.question, k=q.k, model=core.CHAT_MODEL)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn
    print("\n  RAG web UI  ->  http://localhost:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
