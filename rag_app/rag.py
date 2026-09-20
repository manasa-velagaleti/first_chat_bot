"""Ask questions about the documents in docs/ - a RAG pipeline on LangChain.

    python rag_app/rag.py --build            # index the documents (do this first)
    python rag_app/rag.py                    # interactive Q&A
    python rag_app/rag.py --ask "What did he say about jobs?"

How it works:
    index:    docs/*.txt -> 500-char chunks -> OpenAI embeddings -> FAISS on disk
    retrieve: question -> embedding -> nearest chunks by vector distance
    generate: those chunks + the question -> gpt-4.1-mini -> grounded answer

The model is told to answer only from the retrieved chunks and to say so when
the documents don't cover the question. That is the whole point of RAG: the
answer is traceable to a source you can read, rather than to model memory.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

warnings.filterwarnings("ignore")   # langchain-community sunset notice

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DOCS_DIR = ROOT / "docs"
INDEX_DIR = Path(__file__).resolve().parent / "faiss_index"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# "\n" does the real work here: the source documents put one paragraph per
# line with no blank lines between them, so "\n\n" never matches. The later
# separators only come into play for a paragraph longer than CHUNK_SIZE.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

SYSTEM_PROMPT = """You answer questions using only the excerpts provided.

Rules:
- Use only what is in the excerpts. Do not add outside knowledge.
- If the excerpts do not answer the question, say so plainly. Do not guess.
- Cite the excerpts you used by their number, like [1] or [2].
- Quote the document's own wording where it makes the answer clearer.
- Be concise."""

BAR = "=" * 74


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------

def load_documents() -> list[Document]:
    """Read every .txt and .md file in docs/.

    A DirectoryLoader rather than a single hard-coded file, so dropping
    another document into docs/ and re-running --build picks it up.
    """
    docs: list[Document] = []
    for pattern in ("**/*.txt", "**/*.md"):
        loader = DirectoryLoader(
            str(DOCS_DIR),
            glob=pattern,
            # docs/img/ holds screenshots and their caption file - assets, not
            # source material. Indexing them would put noise in the retrieval.
            exclude=["**/img/**"],
            loader_cls=TextLoader,
            # Explicit encoding: without it this fails on Windows, where the
            # default is cp1252 and these documents contain curly quotes.
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        )
        docs.extend(loader.load())

    for d in docs:
        src = Path(d.metadata.get("source", "?"))
        d.metadata["file"] = src.name
    return docs


def build_index(verbose: bool = True) -> FAISS:
    """Split, embed, and write a FAISS index to disk."""
    docs = load_documents()
    if not docs:
        raise SystemExit(f"No .txt or .md files found in {DOCS_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

    # Number the chunks so a retrieved excerpt can be traced back to its
    # position in the source document.
    for i, c in enumerate(chunks):
        c.metadata["chunk"] = i

    if verbose:
        sizes = [len(c.page_content) for c in chunks]
        print(BAR)
        print(f"Documents : {len(docs)} "
              f"({', '.join(sorted({d.metadata['file'] for d in docs}))})")
        print(f"Characters: {sum(len(d.page_content) for d in docs):,}")
        print(f"Chunking  : recursive, {CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap")
        print(f"Separators: {SEPARATORS}")
        print(f"Chunks    : {len(chunks)} "
              f"(smallest {min(sizes)}, largest {max(sizes)}, "
              f"average {sum(sizes) // len(sizes)})")
        print(f"Embedding : {EMBED_MODEL}")
        print(BAR)
        print("Embedding chunks...")

    store = FAISS.from_documents(chunks, OpenAIEmbeddings(model=EMBED_MODEL))

    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    store.save_local(str(INDEX_DIR))

    if verbose:
        print(f"Index saved to {INDEX_DIR.relative_to(ROOT)} "
              f"({store.index.ntotal} vectors)\n")
    return store


def load_index() -> FAISS:
    """Load the index from disk, building it first if it isn't there."""
    if not INDEX_DIR.exists():
        print("No index yet - building one first.\n")
        return build_index()

    return FAISS.load_local(
        str(INDEX_DIR),
        OpenAIEmbeddings(model=EMBED_MODEL),
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
    # Retrieve. The score is L2 distance between embeddings, so LOWER means
    # more similar - it is not a 0-1 similarity score.
    hits = store.similarity_search_with_score(question, k=k)
    if not hits:
        print("Nothing retrieved - is the index empty?")
        return

    context = format_context(hits)

    llm = ChatOpenAI(model=model, temperature=0)
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Excerpts:\n\n{context}\n\nQuestion: {question}"},
    ])

    print(f"\n{response.text}\n")

    if show_sources:
        print("-" * 74)
        print(f"Retrieved {len(hits)} excerpts (distance: lower = closer match)")
        for n, (doc, score) in enumerate(hits, start=1):
            snippet = " ".join(doc.page_content.split())
            if len(snippet) > 100:
                snippet = snippet[:99] + "…"
            print(f"  [{n}] distance {score:.3f} | "
                  f"{doc.metadata.get('file', '?')} chunk "
                  f"{doc.metadata.get('chunk', '?')}")
            print(f"      {snippet}")
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
    ap.add_argument("--model", default=CHAT_MODEL,
                    help=f"OpenAI chat model (default: {CHAT_MODEL})")
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
        print("Try rebuilding:  python rag_app/rag.py --build")
        return 1

    show = not args.no_sources

    if args.ask:
        answer(store, args.ask, args.k, args.model, show)
        return 0

    # Interactive
    print(BAR)
    print(f"RAG over docs/  |  {args.model}  |  top {args.k} chunks")
    print("Ask a question, or 'exit' to quit.")
    print(BAR)

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
