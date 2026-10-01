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

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The RAG code lives one folder over; add it to the path rather than copying
# any of it here.
sys.path.insert(0, str(ROOT / "embeddings_app"))

import core                                    # noqa: E402  (loads .env)
import rag_pinecone as rag                     # noqa: E402

from fastapi import FastAPI, File, UploadFile  # noqa: E402
from fastapi.responses import FileResponse     # noqa: E402
from fastapi.staticfiles import StaticFiles    # noqa: E402
from pydantic import BaseModel, Field          # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
DOCS_DIR = ROOT / "docs"

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


# --- documents ------------------------------------------------------------

ALLOWED = {".txt", ".md", ".pdf"}
MAX_UPLOAD = 25 * 1024 * 1024        # 25 MB
MANIFEST = Path(__file__).resolve().parent / ".built.json"


def corpus_fingerprint() -> str:
    """A hash of the current documents: name, size and mtime of each.

    Recorded when a namespace is built. If it later differs, that namespace
    holds vectors for documents that have since changed - so the UI can say
    "rebuild needed" instead of quietly answering from stale text.
    """
    parts = []
    for p in sorted(DOCS_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in ALLOWED and "img" not in p.parts:
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def read_manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text())
    except Exception:
        return {}


def record_built(namespace: str) -> None:
    m = read_manifest()
    m[namespace] = corpus_fingerprint()
    MANIFEST.write_text(json.dumps(m, indent=2))


def list_documents() -> list[dict]:
    out = []
    for p in sorted(DOCS_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in ALLOWED and "img" not in p.parts:
            out.append({
                "name": p.name,
                "kind": p.suffix.lower().lstrip("."),
                "kb": round(p.stat().st_size / 1024, 1),
            })
    return out


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
    manifest = read_manifest()
    fingerprint = corpus_fingerprint()
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
            # Built, but from a different set of documents than are here now.
            # An unrecorded namespace counts as stale too: it holds vectors of
            # unknown provenance, and silently answering from them is worse
            # than asking for one rebuild.
            "stale": (counts.get(ns, 0) > 0
                      and manifest.get(ns) != fingerprint),
            "vectors": counts.get(ns, 0),
            "would_make": len(texts),
            "smallest": min(sizes),
            "largest": max(sizes),
            "average": sum(sizes) // len(sizes),
            "build_cost": round(cost, 6),
        })
    return {"options": out, "overlap": core.CHUNK_OVERLAP,
            "documents": list_documents()}


@app.post("/api/build")
def build(req: BuildRequest) -> dict:
    """Chunk, embed and upload one setting. Takes tens of seconds."""
    ns = rag.namespace_for(req.chunk_size)
    try:
        rag.build(rag.INDEX_NAME, ns, req.chunk_size, verbose=False)
        _stores.pop(ns, None)          # force a reopen against fresh vectors
        record_built(ns)
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


@app.get("/api/documents")
def documents() -> dict:
    return {"documents": list_documents()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    """Accept a document and drop it into docs/.

    Saved into the same folder the CLIs read, so an uploaded file is picked
    up by rag.py and embed.py too - not just this page.
    """
    raw_name = Path(file.filename or "").name          # strip any path
    suffix = Path(raw_name).suffix.lower()
    if suffix not in ALLOWED:
        return {"error": f"{suffix or 'That file type'} is not supported. "
                         f"Use .txt, .md or .pdf."}

    # Keep the name recognisable but safe to put on disk.
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(raw_name).stem)[:80] or "document"
    target = DOCS_DIR / f"{safe}{suffix}"
    n = 1
    while target.exists():
        target = DOCS_DIR / f"{safe}_{n}{suffix}"
        n += 1

    size = 0
    try:
        with target.open("wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    out.close()
                    target.unlink(missing_ok=True)
                    return {"error": f"File is larger than "
                                     f"{MAX_UPLOAD // (1024*1024)} MB."}
                out.write(chunk)
    except Exception as exc:
        target.unlink(missing_ok=True)
        return {"error": f"{type(exc).__name__}: {exc}"}

    # Confirm it is actually readable before reporting success - a corrupt PDF
    # would otherwise only fail later, during a build.
    try:
        loaded = [d for d in core.load_documents()
                  if d.metadata.get("file") == target.name]
        chars = sum(len(d.page_content) for d in loaded)
        if chars == 0:
            target.unlink(missing_ok=True)
            return {"error": "No text could be extracted. A scanned PDF "
                             "needs OCR before it can be indexed."}
    except Exception as exc:
        target.unlink(missing_ok=True)
        return {"error": f"Could not read the file: {exc}"}

    return {"ok": True, "name": target.name, "kb": round(size / 1024, 1),
            "parts": len(loaded), "characters": chars,
            "documents": list_documents()}


@app.delete("/api/documents/{name}")
def delete_document(name: str) -> dict:
    target = DOCS_DIR / Path(name).name          # no path traversal
    if not target.exists() or target.suffix.lower() not in ALLOWED:
        return {"error": "No such document."}
    target.unlink()
    return {"ok": True, "documents": list_documents()}


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
