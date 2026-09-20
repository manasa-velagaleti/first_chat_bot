"""RAG over docs/ using Pinecone as the vector database.

    python embeddings_app/rag_pinecone.py --build      # create index + upload
    python embeddings_app/rag_pinecone.py              # interactive Q&A
    python embeddings_app/rag_pinecone.py --ask "What about jobs?"
    python embeddings_app/rag_pinecone.py --stats      # what's in the index

Same pipeline as rag.py, with two deliberate differences:

    rag.py           FAISS on disk   | fixed 500-character chunks
    rag_pinecone.py  Pinecone cloud  | one chunk per paragraph

Paragraph chunking keeps each idea whole instead of cutting at a character
count, so a retrieved excerpt is a complete thought. The trade is uneven
chunk sizes - in this document, 20 to 1,172 characters.

Needs PINECONE_API_KEY in .env alongside OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# core first: it silences the langchain-community sunset warning, which fires
# at import time.
import core

from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = "sotu-paragraphs"
DIMENSION = 1536            # text-embedding-3-small
METRIC = "cosine"
CLOUD, REGION = "aws", "us-east-1"      # free tier

SYSTEM_PROMPT = """You answer questions using only the excerpts provided.

Rules:
- Use only what is in the excerpts. Do not add outside knowledge.
- If the excerpts do not answer the question, say so plainly. Do not guess.
- Cite the excerpts you used by their number, like [1] or [2].
- Quote the document's own wording where it makes the answer clearer.
- Be concise."""


def client() -> Pinecone:
    key = os.getenv("PINECONE_API_KEY")
    if not key:
        raise SystemExit(
            "PINECONE_API_KEY is not set.\n"
            "Get one free at https://app.pinecone.io and add it to .env, "
            "then re-run."
        )
    return Pinecone(api_key=key)


def ensure_index(pc: Pinecone, name: str, verbose: bool = True):
    """Create the index if it doesn't exist, and wait until it's usable."""
    if pc.has_index(name):
        if verbose:
            print(f"Index '{name}' already exists.")
        return pc.Index(name)

    if verbose:
        print(f"Creating index '{name}' "
              f"({DIMENSION} dims, {METRIC}, {CLOUD}/{REGION})...")
    pc.create_index(
        name=name,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )

    # Creation is asynchronous; upserting before it is ready fails.
    while not pc.describe_index(name).status["ready"]:
        time.sleep(1)
    if verbose:
        print("Index ready.")
    return pc.Index(name)


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def build(name: str, namespace: str, verbose: bool = True) -> PineconeVectorStore:
    docs = core.load_documents()
    if not docs:
        raise SystemExit(f"No .txt or .md files found in {core.DOCS_DIR}")

    chunks = core.split_documents_by_paragraph(docs)

    if verbose:
        texts = [c.page_content for c in chunks]
        tokens, cost = core.estimate_cost(texts)
        split_up = sum(1 for c in chunks if "part" in c.metadata)
        print(core.BAR)
        print(f"Documents : {len(docs)} "
              f"({', '.join(sorted({d.metadata['file'] for d in docs}))})")
        print(f"Chunking  : by paragraph "
              f"(split only above {core.PARAGRAPH_MAX} chars)")
        print(f"Chunks    : {core.chunk_stats(texts)}")
        print(f"          : {split_up} oversized paragraph(s) had to be split")
        print(f"Embedding : {core.EMBED_MODEL} ({DIMENSION} dims)")
        print(f"Tokens    : {tokens:,}  (about ${cost:.6f})")
        print(f"Vector DB : Pinecone index '{name}'"
              + (f", namespace '{namespace}'" if namespace else ""))
        print(core.BAR)

    pc = client()
    ensure_index(pc, name, verbose)

    # Clear any previous run so --build is a true rebuild rather than a
    # second copy of every chunk sitting alongside the first.
    index = pc.Index(name)
    try:
        stats = index.describe_index_stats()
        existing = stats.get("namespaces", {}).get(namespace or "", {})
        if existing.get("vector_count", 0):
            if verbose:
                print(f"Clearing {existing['vector_count']} existing vectors...")
            index.delete(delete_all=True, namespace=namespace or None)
            time.sleep(2)
    except Exception:
        pass        # empty index - nothing to clear

    if verbose:
        print("Embedding and uploading...")

    store = PineconeVectorStore.from_documents(
        chunks,
        embedding=core.embeddings(),
        index_name=name,
        namespace=namespace or None,
    )

    if verbose:
        # Upserts are eventually consistent, so the count can lag a moment.
        time.sleep(3)
        total = index.describe_index_stats().get("total_vector_count", "?")
        print(f"Uploaded. Index now holds {total} vectors.\n")
    return store


def open_store(name: str, namespace: str) -> PineconeVectorStore:
    pc = client()
    if not pc.has_index(name):
        raise SystemExit(
            f"Index '{name}' does not exist.\n"
            f"Build it first:  python embeddings_app/rag_pinecone.py --build"
        )
    return PineconeVectorStore.from_existing_index(
        index_name=name,
        embedding=core.embeddings(),
        namespace=namespace or None,
    )


def show_stats(name: str) -> None:
    pc = client()
    if not pc.has_index(name):
        print(f"Index '{name}' does not exist yet.")
        return
    desc = pc.describe_index(name)
    stats = pc.Index(name).describe_index_stats()
    print(core.BAR)
    print(f"Index      : {name}")
    print(f"Dimension  : {desc.dimension}   Metric: {desc.metric}")
    print(f"Host       : {desc.host}")
    print(f"Vectors    : {stats.get('total_vector_count', 0)}")
    for ns, info in (stats.get("namespaces") or {}).items():
        print(f"  namespace '{ns or '(default)'}': {info.get('vector_count', 0)}")
    print(core.BAR)


# --------------------------------------------------------------------------
# Ask
# --------------------------------------------------------------------------

def answer(store: PineconeVectorStore, question: str, k: int, model: str,
           show_sources: bool) -> None:
    # Pinecone is configured with cosine similarity, so HIGHER is better here
    # - the opposite of the L2 distance FAISS returns in rag.py.
    hits = store.similarity_search_with_score(question, k=k)
    if not hits:
        print("Nothing retrieved - is the index empty? Try --stats.")
        return

    context = "\n\n".join(
        f"[{n}] (from {doc.metadata.get('file', '?')})\n{doc.page_content}"
        for n, (doc, _score) in enumerate(hits, start=1)
    )

    llm = ChatOpenAI(model=model, temperature=0)
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Excerpts:\n\n{context}\n\n"
                                    f"Question: {question}"},
    ])

    print(f"\n{response.text}\n")

    if show_sources:
        print("-" * 74)
        print(f"Retrieved {len(hits)} excerpts "
              f"(cosine similarity: higher = closer match)")
        for n, (doc, score) in enumerate(hits, start=1):
            # Pinecone stores metadata numbers as floats, so an int that went
            # in as 33 comes back as 33.0 - cast it before displaying.
            para = doc.metadata.get("paragraph")
            where = f"paragraph {int(para)}" if para is not None else "paragraph ?"
            part = doc.metadata.get("part")
            if part:
                where += f" part {part}"
            print(f"  [{n}] score {score:.3f} | "
                  f"{doc.metadata.get('file', '?')} {where}")
            print(f"      {core.preview(doc.page_content, 100)}")
        print("-" * 74)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--build", action="store_true",
                    help="create the index and upload docs/, then exit")
    ap.add_argument("--ask", help="ask one question and exit")
    ap.add_argument("--stats", action="store_true",
                    help="show what the index contains, then exit")
    ap.add_argument("--index", default=INDEX_NAME,
                    help=f"Pinecone index name (default: {INDEX_NAME})")
    ap.add_argument("--namespace", default="",
                    help="Pinecone namespace - lets one index hold several "
                         "separate document sets")
    ap.add_argument("-k", type=int, default=4,
                    help="how many chunks to retrieve (default: 4)")
    ap.add_argument("--model", default=core.CHAT_MODEL,
                    help=f"OpenAI chat model (default: {core.CHAT_MODEL})")
    ap.add_argument("--no-sources", action="store_true",
                    help="hide the retrieved excerpts")
    args = ap.parse_args()

    if args.stats:
        show_stats(args.index)
        return 0

    if args.build:
        build(args.index, args.namespace)
        return 0

    try:
        store = open_store(args.index, args.namespace)
    except SystemExit:
        raise
    except Exception as err:
        print(f"Could not open the index: {type(err).__name__}: {err}")
        return 1

    show = not args.no_sources

    if args.ask:
        answer(store, args.ask, args.k, args.model, show)
        return 0

    print(core.BAR)
    print(f"Pinecone RAG  |  index '{args.index}'  |  {args.model}  "
          f"|  top {args.k}")
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
