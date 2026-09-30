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

# The chunk sizes offered in the UI. None means paragraph chunking.
# Each one lives in its own Pinecone namespace, so a setting you have already
# built stays built and switching back to it needs no re-embedding.
CHUNK_OPTIONS: list[int | None] = [None, 300, 500, 700, 1000]

# Opening a store creates an embeddings client and a Pinecone connection, so
# they are cached per namespace rather than rebuilt per request.
_stores: dict[str, object] = {}


def store(namespace: str):
    if namespace not in _stores:
        _stores[namespace] = rag.open_store(rag.INDEX_NAME, namespace)
    return _stores[namespace]


def namespace_counts() -> dict[str, int]:
    """Vector count per namespace - tells the UI which settings are built."""
    try:
        pc = rag.client()
        if not pc.has_index(rag.INDEX_NAME):
            return {}
        stats = pc.Index(rag.INDEX_NAME).describe_index_stats()
        return {ns: info.get("vector_count", 0)
                for ns, info in (stats.get("namespaces") or {}).items()}
    except Exception:
        return {}


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)
    chunk_size: int | None = Field(default=None)


class BuildRequest(BaseModel):
    chunk_size: int | None = Field(default=None)


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
        }
    except SystemExit as exc:
        return {"ready": False, "error": str(exc)}
    except Exception as exc:
        return {"ready": False, "error": f"{type(exc).__name__}: {exc}"}


@app.get("/api/configs")
def configs() -> dict:
    """Every chunk setting, and how many vectors each already holds."""
    counts = namespace_counts()
    docs = core.load_documents()
    out = []
    for size in CHUNK_OPTIONS:
        ns = rag.namespace_for(size)
        # Chunking is local and cheap, so the UI can show what a setting would
        # produce before anyone pays to embed it.
        texts = [c.page_content for c in rag.chunk_for(docs, size)]
        sizes = [len(t) for t in texts] or [0]
        tokens, cost = core.estimate_cost(texts)
        out.append({
            "chunk_size": size,
            "label": "By paragraph" if size is None else f"{size} characters",
            "namespace": ns,
            "built": counts.get(ns, 0) > 0,
            "vectors": counts.get(ns, 0),
            "would_make": len(texts),
            "smallest": min(sizes),
            "largest": max(sizes),
            "average": sum(sizes) // len(sizes),
            "build_cost": round(cost, 6),
        })
    return {"options": out, "overlap": core.CHUNK_OVERLAP}


@app.post("/api/build")
def build(req: BuildRequest) -> dict:
    """Chunk, embed and upload one setting. Takes tens of seconds."""
    ns = rag.namespace_for(req.chunk_size)
    try:
        rag.build(rag.INDEX_NAME, ns, req.chunk_size, verbose=False)
        _stores.pop(ns, None)          # force a reopen against fresh vectors
        return {"ok": True, "namespace": ns,
                "vectors": namespace_counts().get(ns, 0)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@app.post("/api/ask")
def ask(q: Question) -> dict:
    ns = rag.namespace_for(q.chunk_size)
    if namespace_counts().get(ns, 0) == 0:
        return {"error": f"Nothing indexed for this chunk setting yet. "
                         f"Open the chunking dialog and build it first."}
    try:
        # The CLI's own function - no retrieval or prompting logic here.
        result = rag.ask(store(ns), q.question, k=q.k, model=core.CHAT_MODEL)
        result["chunking"] = ("by paragraph" if q.chunk_size is None
                              else f"{q.chunk_size} characters")
        return result
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


@app.get("/")
def index() -> FileResponse:
    # no-store: this is a local dev tool and the page is edited constantly.
    # Without it the browser can serve a cached copy without revalidating,
    # so an edit appears to have had no effect.
    return FileResponse(
        STATIC / "index.html",
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn
    print("\n  RAG web UI  ->  http://localhost:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
