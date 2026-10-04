"""Document search, exposed as a tool the chat agent can choose to call.

The retrieval itself is not implemented here - it is rag_pinecone.ask(), the
same function the CLI and the web page use. This module only wraps it in the
shape LangChain wants and remembers what came back, so the UI can show sources.

Why a tool rather than retrieving before every message: most turns in a
conversation are not document questions. "Thanks, that helps" should not trigger
a search, and forcing one drags irrelevant passages into the context. Letting
the model decide keeps ordinary chat ordinary.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "embeddings_app"))

import core                       # noqa: E402  (loads .env)
import rag_pinecone as rag        # noqa: E402

from langchain_core.tools import tool      # noqa: E402

# What the last search returned, per conversation. app.py reads this to render
# the sources panel; the agent itself gets the text through the tool's return.
_LAST_SOURCES: dict[str, list[dict]] = {}

# Opening a store builds an embeddings client and a Pinecone connection, so
# they are kept per namespace rather than rebuilt on every question.
_STORES: dict[str, object] = {}


def last_sources(thread_id: str) -> list[dict]:
    return _LAST_SOURCES.get(thread_id, [])


def clear_sources(thread_id: str) -> None:
    _LAST_SOURCES.pop(thread_id, None)


def _store(namespace: str):
    if namespace not in _STORES:
        _STORES[namespace] = rag.open_store(rag.INDEX_NAME, namespace)
    return _STORES[namespace]


def drop_store(namespace: str) -> None:
    """Forget a cached store - call after re-indexing that namespace."""
    _STORES.pop(namespace, None)


def describe_corpus(namespace: str) -> str:
    """Names of the documents in a namespace, for the tool description.

    The docstring is the model's only guide to whether a question is worth
    searching. "Search the documents" is too vague to judge against; naming the
    actual files lets it tell a document question from a general one.
    """
    try:
        pc = rag.client()
        if not pc.has_index(rag.INDEX_NAME):
            return ""
        index = pc.Index(rag.INDEX_NAME)
        names: set[str] = set()
        # A sample is enough - this only has to name the documents, and
        # listing a large namespace in full would be slow for no gain.
        for page in index.list(namespace=namespace, limit=100):
            got = index.fetch(ids=list(page), namespace=namespace)
            for v in got.vectors.values():
                f = (v.metadata or {}).get("file")
                if f:
                    names.add(f)
            break
        return ", ".join(sorted(names))
    except Exception:
        return ""


def make_search_tool(thread_id: str, namespaces, k: int = 4):
    """Build the search tool for one conversation.

    A factory rather than a module-level tool because the thread id, namespaces
    and k all vary per conversation, and a @tool function takes only the
    arguments the model supplies.

    namespaces may be one name or several. Several is the normal case: a chat
    searches the documents shared across the app *and* any attached to that
    chat alone. Pinecone queries one namespace at a time, so each is searched
    and the results merged on score - they are comparable because every
    namespace is embedded with the same model.
    """
    if isinstance(namespaces, str):
        namespaces = [namespaces]
    namespaces = [n for n in namespaces if n]

    corpus = ", ".join(filter(None, (describe_corpus(n) for n in namespaces)))
    about = f" The documents available are: {corpus}." if corpus else ""

    @tool
    def search_documents(query: str) -> str:
        """Search the user's uploaded documents and return relevant passages.

        Use this whenever a question might be answered by the documents - their
        contents, specific figures, names, dates, requirements or wording. Also
        use it when the user refers to "the document", "the file", "the report"
        or similar. Prefer searching over guessing.

        Do not use it for general knowledge, for chitchat, or for questions
        about the conversation itself.

        Args:
            query: What to look for, phrased as the user would describe it.
        """
        hits = []
        for ns in namespaces:
            try:
                hits.extend(_store(ns).similarity_search_with_score(query, k=k))
            except Exception as exc:
                # One namespace failing should not lose the others.
                if len(namespaces) == 1:
                    return (f"The document search is unavailable: "
                            f"{type(exc).__name__}: {exc}")
        # Cosine similarity: higher is closer, so the best come first.
        hits.sort(key=lambda h: h[1], reverse=True)
        hits = hits[:k]

        if not hits:
            return "No passages in the documents matched that."

        sources = []
        for n, (doc, score) in enumerate(hits, start=1):
            meta = doc.metadata or {}
            page = meta.get("page")
            para = meta.get("paragraph")
            sources.append({
                "n": n,
                "score": round(float(score), 3),
                "file": meta.get("file", "?"),
                "page": int(page) if page is not None else None,
                "paragraph": int(para) if para is not None else None,
                "text": doc.page_content,
            })
        _LAST_SOURCES[thread_id] = sources

        return "\n\n".join(
            f"[{s['n']}] ({s['file']}"
            + (f" page {s['page']}" if s["page"] is not None else "")
            + f")\n{s['text']}"
            for s in sources
        )

    # Append the document names to the description the model actually sees.
    if about:
        search_documents.description = search_documents.description + about
    return search_documents


def all_chunks(namespaces, limit: int = 2000) -> list[dict]:
    """Every chunk stored in these namespaces, in document order.

    Pinecone has no "list everything" query - list() gives ids and fetch()
    turns ids into records, so this pages through both. Fine for a corpus of
    this size; a very large index would want a cursor rather than a cap.

    This is what "show me the chunks" actually means. The search tool returns
    the best k for a question, which is a different thing and cannot answer
    "how was this document cut up?".
    """
    if isinstance(namespaces, str):
        namespaces = [namespaces]

    out: list[dict] = []
    try:
        pc = rag.client()
        if not pc.has_index(rag.INDEX_NAME):
            return []
        index = pc.Index(rag.INDEX_NAME)
    except Exception:
        return []

    for ns in [n for n in namespaces if n]:
        ids: list[str] = []
        try:
            for page in index.list(namespace=ns, limit=100):
                ids.extend(page)
                if len(ids) >= limit:
                    break
        except Exception:
            continue

        for i in range(0, len(ids), 100):
            try:
                got = index.fetch(ids=ids[i:i + 100], namespace=ns)
            except Exception:
                continue
            for vec in got.vectors.values():
                meta = vec.metadata or {}
                out.append({
                    "file": meta.get("file", "?"),
                    "page": int(meta["page"]) if meta.get("page") is not None else None,
                    "chunk": int(meta["chunk"]) if meta.get("chunk") is not None else None,
                    "text": meta.get("text", ""),
                    "namespace": ns,
                })

    # Document order, so the list reads the way the document does.
    out.sort(key=lambda c: (c["file"], c["page"] if c["page"] is not None else 0,
                            c["chunk"] if c["chunk"] is not None else 0))
    return out
