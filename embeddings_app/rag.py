"""Ask questions about docs/ - RAG built on the same chunking as embed.py.

    python embeddings_app/rag.py --build          # index docs/ (do this first)
    python embeddings_app/rag.py                  # interactive Q&A
    python embeddings_app/rag.py --ask "What did he say about jobs?"

This is embed.py's pipeline with two stages added on the end:

    embed.py :  load -> chunk -> embed -> show the vectors
    rag.py   :  load -> chunk -> embed -> store in FAISS
                                       -> retrieve nearest -> generate answer

Loading, chunking and the embedding model all come from core.py, so a chunk
you inspected with embed.py is the same chunk retrieved here.

The model is told to answer only from the retrieved chunks and to say so when
they don't cover the question. That is the point of RAG: the answer traces to
a passage you can read, rather than to model memory.
"""

from __future__ import annotations

import argparse
import shutil
import sys

# core first: it silences the langchain-community sunset warning, which is
# emitted at import time - so importing FAISS above this line would leak it.
import core

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """You answer questions using only the excerpts provided.

Rules:
- Use only what is in the excerpts. Do not add outside knowledge.
- If the excerpts do not answer the question, say so plainly. Do not guess.
- Cite the excerpts you used by their number, like [1] or [2].
- Quote the document's own wording where it makes the answer clearer.
- Be concise."""


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------

def build_index(verbose: bool = True) -> FAISS:
    """Split, embed, and write a FAISS index to disk."""
    docs = core.load_documents()
    if not docs:
        raise SystemExit(f"No .txt or .md files found in {core.DOCS_DIR}")

    chunks = core.split_documents(docs)

    if verbose:
        names = ", ".join(sorted({d.metadata["file"] for d in docs}))
        texts = [c.page_content for c in chunks]
        tokens, cost = core.estimate_cost(texts)
        print(core.BAR)
        print(f"Documents : {len(docs)} ({names})")
        print(f"Characters: {sum(len(d.page_content) for d in docs):,}")
        print(f"Chunking  : recursive, {core.CHUNK_SIZE} chars, "
              f"{core.CHUNK_OVERLAP} overlap")
        print(f"Separators: {core.SEPARATORS}")
        print(f"Chunks    : {core.chunk_stats(texts)}")
        print(f"Embedding : {core.EMBED_MODEL}")
        print(f"Tokens    : {tokens:,}  (about ${cost:.6f})")
        print(core.BAR)
        print("Embedding chunks...")

    store = FAISS.from_documents(chunks, core.embeddings())

    if core.INDEX_DIR.exists():
        shutil.rmtree(core.INDEX_DIR)
    store.save_local(str(core.INDEX_DIR))

    if verbose:
        print(f"Index saved to {core.INDEX_DIR.relative_to(core.ROOT)} "
              f"({store.index.ntotal} vectors)\n")
    return store


def load_index() -> FAISS:
    """Load the index from disk, building it first if it isn't there."""
    if not core.INDEX_DIR.exists():
        print("No index yet - building one first.\n")
        return build_index()

    return FAISS.load_local(
        str(core.INDEX_DIR),
        core.embeddings(),
        # FAISS metadata is pickle-backed, so loading executes code. Safe here
        # because this index was written by build_index() on this machine;
        # never point this at an index you downloaded from somewhere.
        allow_dangerous_deserialization=True,
    )


# --------------------------------------------------------------------------
# Retrieval + generation
# --------------------------------------------------------------------------

def format_context(hits: list[tuple[Document, float]]) -> str:
    """Number the excerpts so the model can cite them."""
    return "\n\n".join(
        f"[{n}] (from {doc.metadata.get('file', '?')})\n{doc.page_content}"
        for n, (doc, _score) in enumerate(hits, start=1)
    )


def answer(store: FAISS, question: str, k: int, model: str,
           show_sources: bool) -> None:
    # The score is L2 distance between embeddings, so LOWER means more
    # similar - it is not a 0-1 similarity score.
    hits = store.similarity_search_with_score(question, k=k)
    if not hits:
        print("Nothing retrieved - is the index empty?")
        return

    llm = ChatOpenAI(model=model, temperature=0)
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Excerpts:\n\n{format_context(hits)}\n\n"
                    f"Question: {question}"},
    ])

    print(f"\n{response.text}\n")

    if show_sources:
        print("-" * 74)
        print(f"Retrieved {len(hits)} excerpts (distance: lower = closer match)")
        for n, (doc, score) in enumerate(hits, start=1):
            print(f"  [{n}] distance {score:.3f} | "
                  f"{doc.metadata.get('file', '?')} chunk "
                  f"{doc.metadata.get('chunk', '?')}")
            print(f"      {core.preview(doc.page_content, 100)}")
        print("-" * 74)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--build", action="store_true",
                    help="(re)build the index from docs/, then exit")
    ap.add_argument("--ask", help="ask one question and exit")
    ap.add_argument("-k", type=int, default=4,
                    help="how many chunks to retrieve (default: 4)")
    ap.add_argument("--model", default=core.CHAT_MODEL,
                    help=f"OpenAI chat model (default: {core.CHAT_MODEL})")
    ap.add_argument("--no-sources", action="store_true",
                    help="hide the retrieved excerpts")
    args = ap.parse_args()

    if args.build:
        build_index()
        return 0

    try:
        store = load_index()
    except Exception as err:
        print(f"Could not load the index: {type(err).__name__}: {err}")
        print("Try rebuilding:  python embeddings_app/rag.py --build")
        return 1

    show = not args.no_sources

    if args.ask:
        answer(store, args.ask, args.k, args.model, show)
        return 0

    print(core.BAR)
    print(f"RAG over docs/  |  {args.model}  |  top {args.k} chunks")
    print("Ask a question, or 'exit' to quit.")
    print(core.BAR)

    while True:
        try:
            q = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye.")
            return 0
        try:
            answer(store, q, args.k, args.model, show)
        except Exception as err:
            print(f"\n[error: {type(err).__name__}: {err}]")


if __name__ == "__main__":
    sys.exit(main())
