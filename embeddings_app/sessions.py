"""Scratch corpora: documents that exist only for one session.

A session owns a temp folder and a Pinecone namespace. Files uploaded into it
never touch the project, and clearing it removes both. The point is being able
to ask questions about a document without committing it to anything.

Shared by the Streamlit chatbot and the FastAPI web app. Written once rather
than twice: the guarantees below are the whole feature, and two copies would
drift until one of them quietly stopped honouring them.
"""

from __future__ import annotations

import atexit
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import core
import rag_pinecone as rag
from langchain_pinecone import PineconeVectorStore

MAX_UPLOAD = 25 * 1024 * 1024          # 25 MB
DEFAULT_CHUNK = 700                    # fixed-size suits PDFs, which most are


@dataclass
class Session:
    id: str
    dir: Path
    namespace: str
    chunk_size: int = DEFAULT_CHUNK
    files: list[dict] = field(default_factory=list)   # {name, kb, chunks, characters}

    @property
    def names(self) -> list[str]:
        return [f["name"] for f in self.files]


SESSIONS: dict[str, Session] = {}


def start(chunk_size: int = DEFAULT_CHUNK) -> Session:
    sid = uuid.uuid4().hex
    sess = Session(
        id=sid,
        dir=Path(tempfile.mkdtemp(prefix=f"ragsess_{sid[:8]}_")),
        namespace=f"sess-{sid[:12]}",
        chunk_size=chunk_size,
    )
    SESSIONS[sid] = sess
    return sess


def get(sid: str | None) -> Session | None:
    return SESSIONS.get(sid) if sid else None


def clear(sid: str) -> None:
    """Delete the session's vectors and its temp files. Safe to call twice."""
    sess = SESSIONS.pop(sid, None)
    if not sess:
        return
    try:
        pc = rag.client()
        if pc.has_index(rag.INDEX_NAME):
            pc.Index(rag.INDEX_NAME).delete(delete_all=True,
                                            namespace=sess.namespace)
    except Exception:
        pass            # namespace may never have been written to
    shutil.rmtree(sess.dir, ignore_errors=True)


@atexit.register
def cleanup_all() -> None:
    """Never leave temp folders or stray namespaces behind on shutdown."""
    for sid in list(SESSIONS):
        clear(sid)


def _safe_target(sess: Session, filename: str) -> Path:
    """A writable path inside the session folder, derived from the given name."""
    raw = Path(filename or "").name                 # strip any directory part
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(raw).stem)[:80] or "document"
    suffix = Path(raw).suffix.lower()
    target = sess.dir / f"{stem}{suffix}"
    n = 1
    while target.exists():
        target = sess.dir / f"{stem}_{n}{suffix}"
        n += 1
    return target


def add_document(sess: Session, filename: str, data: bytes) -> dict:
    """Save, read, chunk and embed one file - all before returning.

    One step rather than upload-then-build: a scratch document is only useful
    once it is searchable, so there is nothing to decide in between. Returns
    {"ok": True, ...} or {"error": "..."}; it never raises at the caller.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in core.SUPPORTED:
        return {"error": f"{suffix or 'That file type'} is not supported. "
                         f"Use {', '.join(core.SUPPORTED)}."}
    if len(data) > MAX_UPLOAD:
        return {"error": f"File is larger than {MAX_UPLOAD // (1024*1024)} MB."}

    target = _safe_target(sess, filename)
    try:
        target.write_bytes(data)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    try:
        docs = core.load_one(target)
        chars = sum(len(d.page_content) for d in docs)
        if chars == 0:
            detail = (core.describe_pdf(target) if suffix == ".pdf"
                      else "the file appears to be empty")
            target.unlink(missing_ok=True)
            return {"error": f"No text could be read from that file - {detail}."}

        chunks = rag.chunk_for(docs, sess.chunk_size)
        PineconeVectorStore.from_documents(
            chunks,
            embedding=core.embeddings(),
            index_name=rag.INDEX_NAME,
            namespace=sess.namespace,
        )
    except Exception as exc:
        target.unlink(missing_ok=True)
        return {"error": f"Could not index that file: {type(exc).__name__}: {exc}"}

    entry = {"name": target.name, "kb": round(len(data) / 1024, 1),
             "chunks": len(chunks), "characters": chars}
    sess.files.append(entry)
    return {"ok": True, **entry}


def remove_document(sess: Session, name: str) -> dict:
    """Drop one file from the session.

    Its vectors stay in the namespace until the session is cleared - Pinecone
    cannot delete by metadata on a serverless index without a fetch-and-delete
    pass, and the whole namespace is discarded shortly anyway. The file stops
    being listed, which is what the caller is really asking for.
    """
    target = sess.dir / Path(name).name
    target.unlink(missing_ok=True)
    sess.files = [f for f in sess.files if f["name"] != name]
    return {"ok": True}
