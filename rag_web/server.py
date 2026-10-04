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

import atexit
import hashlib
import json
import re
import shutil
import sys
import tempfile
import uuid
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
from langchain_pinecone import PineconeVectorStore  # noqa: E402
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
    # When present, search only what this session uploaded.
    session: str | None = Field(default=None)


class BuildRequest(BaseModel):
    chunk_size: int | None = Field(default=None)


# --- documents ------------------------------------------------------------

# Whatever core knows how to read - no second list to keep in sync.
ALLOWED = set(core.SUPPORTED)
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


# --- sessions -------------------------------------------------------------
# A session is a scratch corpus: files land in an OS temp folder rather than
# the project, and their vectors go in a Pinecone namespace of their own.
# Nothing a session creates outlives it - clear_session() removes both.

SESSIONS: dict[str, dict] = {}
SESSION_CHUNK = 700          # fixed-size suits PDFs, which most uploads are


def session_state(sid: str) -> dict:
    if sid not in SESSIONS:
        SESSIONS[sid] = {
            "dir": Path(tempfile.mkdtemp(prefix=f"ragsess_{sid[:8]}_")),
            "namespace": f"sess-{sid[:12]}",
            "files": [],            # [{name, kb, chunks}]
            "chunk_size": SESSION_CHUNK,
        }
    return SESSIONS[sid]


def clear_session(sid: str) -> None:
    """Delete the session's vectors and its temp files."""
    st = SESSIONS.pop(sid, None)
    if not st:
        return
    try:
        pc = rag.client()
        if pc.has_index(rag.INDEX_NAME):
            pc.Index(rag.INDEX_NAME).delete(delete_all=True,
                                            namespace=st["namespace"])
    except Exception:
        pass        # namespace may never have been written to
    _stores.pop(st["namespace"], None)
    shutil.rmtree(st["dir"], ignore_errors=True)


@atexit.register
def _cleanup_all_sessions() -> None:
    """Don't leave temp folders or Pinecone namespaces behind on shutdown."""
    for sid in list(SESSIONS):
        clear_session(sid)


def session_documents(sid: str) -> list[dict]:
    return session_state(sid)["files"] if sid in SESSIONS else []


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
            "documents": list_documents(),
            "accepts": sorted(ALLOWED)}


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
    # A session searches only its own uploads, so the staleness check below
    # does not apply - its namespace is written and read in the same session.
    if q.session:
        if q.session not in SESSIONS:
            return {"error": "That session has ended. Reload the page to "
                             "start a new one."}
        st = SESSIONS[q.session]
        if not st["files"]:
            return {"error": "No documents in this session yet - upload one "
                             "to ask about it."}
        try:
            result = rag.ask(store(st["namespace"]), q.question,
                             k=q.k, model=core.CHAT_MODEL)
            result["chunking"] = f"{st['chunk_size']} characters (session)"
            return result
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    ns = rag.namespace_for(q.chunk_size)
    label = "paragraph" if q.chunk_size is None else f"{q.chunk_size}-character"

    if namespace_counts().get(ns, 0) == 0:
        return {"error": f"Nothing is indexed for the {label} setting yet. "
                         f"Open ⚙ Chunks and build it first."}

    # Refuse rather than answer from vectors built out of different documents.
    # Answering anyway produces a confident reply drawn from the wrong corpus,
    # which is far harder to spot than an error.
    if read_manifest().get(ns) != corpus_fingerprint():
        return {"error": f"The {label} index was built from a different set of "
                         f"documents, so it would not see your newest upload. "
                         f"Rebuild it first - the banner above has a button, or "
                         f"pick an index that is already up to date in "
                         f"⚙ Chunks."}
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
    return {"documents": list_documents(),
            "accepts": sorted(ALLOWED)}


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
                         f"Use {', '.join(sorted(ALLOWED))}."}

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
        loaded = core.load_one(target)
        chars = sum(len(d.page_content) for d in loaded)
        if chars == 0:
            detail = (core.describe_pdf(target)
                      if suffix == ".pdf" else "the file appears to be empty")
            target.unlink(missing_ok=True)
            return {"error": f"No text could be read from that file - {detail}. "
                             f"If it is a scan, a higher-resolution or "
                             f"straighter copy usually works."}
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


# --- session API ----------------------------------------------------------

@app.post("/api/session")
def new_session() -> dict:
    """Start a scratch corpus. The id is held by the browser tab."""
    sid = uuid.uuid4().hex
    st = session_state(sid)
    return {"session": sid, "namespace": st["namespace"],
            "chunk_size": st["chunk_size"]}


@app.get("/api/session/{sid}")
def session_info(sid: str) -> dict:
    if sid not in SESSIONS:
        return {"exists": False, "files": [], "vectors": 0}
    st = SESSIONS[sid]
    return {
        "exists": True,
        "files": st["files"],
        "chunk_size": st["chunk_size"],
        "vectors": namespace_counts().get(st["namespace"], 0),
    }


@app.delete("/api/session/{sid}")
def end_session(sid: str) -> dict:
    clear_session(sid)
    return {"ok": True}


@app.post("/api/session/{sid}/upload")
async def session_upload(sid: str, file: UploadFile = File(...)) -> dict:
    """Save to the session's temp folder and embed it straight away.

    One step rather than upload-then-build: a scratch document is only
    useful once it is searchable, so there is nothing to decide in between.
    """
    st = session_state(sid)

    raw_name = Path(file.filename or "").name
    suffix = Path(raw_name).suffix.lower()
    if suffix not in ALLOWED:
        return {"error": f"{suffix or 'That file type'} is not supported. "
                         f"Use {', '.join(sorted(ALLOWED))}."}

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(raw_name).stem)[:80] or "document"
    target = st["dir"] / f"{safe}{suffix}"
    n = 1
    while target.exists():
        target = st["dir"] / f"{safe}_{n}{suffix}"
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

    # Read it, chunk it, embed it - all before replying, so a success here
    # means the document is genuinely ready to be asked about.
    try:
        docs = core.load_one(target)
        chars = sum(len(d.page_content) for d in docs)
        if chars == 0:
            detail = (core.describe_pdf(target)
                      if suffix == ".pdf" else "the file appears to be empty")
            target.unlink(missing_ok=True)
            return {"error": f"No text could be read from that file - {detail}."}

        chunks = rag.chunk_for(docs, st["chunk_size"])
        PineconeVectorStore.from_documents(
            chunks,
            embedding=core.embeddings(),
            index_name=rag.INDEX_NAME,
            namespace=st["namespace"],
        )
        _stores.pop(st["namespace"], None)      # reopen against new vectors
    except Exception as exc:
        target.unlink(missing_ok=True)
        return {"error": f"Could not index that file: {type(exc).__name__}: {exc}"}

    st["files"].append({"name": target.name, "kb": round(size / 1024, 1),
                        "chunks": len(chunks), "characters": chars})
    return {"ok": True, "name": target.name, "chunks": len(chunks),
            "characters": chars, "files": st["files"]}


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
