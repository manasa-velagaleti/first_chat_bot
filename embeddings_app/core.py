"""Shared foundation for the embedding tools.

Both CLIs in this folder do the same first three steps - find the API key,
load documents, split them the same way - and then diverge:

    embed.py   stops after embedding and shows you the vectors
    rag.py     stores the vectors in FAISS and answers questions with them

Keeping that common part here means the two can't drift apart. If chunking
changed in one but not the other, a chunk you inspected with embed.py would
not be the chunk rag.py actually retrieved, which would make the inspection
tool lie about the system it is meant to explain.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from dotenv import load_dotenv

warnings.filterwarnings("ignore")   # langchain-community sunset notice

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- paths ----------------------------------------------------------------
# These tools live in a subfolder; the API key and documents live above it.
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
DOCS_DIR = ROOT / "docs"
INDEX_DIR = APP_DIR / "faiss_index"
OUT_DIR = APP_DIR / "out"

load_dotenv(ROOT / ".env")

# --- models ---------------------------------------------------------------
EMBED_MODEL = "text-embedding-3-small"          # 1536 dimensions
CHAT_MODEL = "gpt-4.1-mini"
EMBED_PRICE_PER_1M_TOKENS = 0.02                # USD

# --- chunking -------------------------------------------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Tried in order; the splitter only drops to the next when a piece is still
# over CHUNK_SIZE.
#
# "\n" is the one that matters for these documents. The SOTU address puts one
# paragraph per line with no blank lines between them - 106 single newlines
# and zero "\n\n" - so the library's default list, which leads with "\n\n",
# would never match and the split would fall straight through to sentence
# boundaries, cutting paragraphs apart.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# --- paragraph chunking ---------------------------------------------------
# A different strategy, used by rag_pinecone.py: one chunk per paragraph
# rather than a fixed character budget.
#
# The two differ in what they optimise for. Fixed-size chunking gives evenly
# sized vectors and predictable cost, but cuts a long paragraph in half and
# glues short ones together. Paragraph chunking keeps each idea whole, at the
# price of wildly uneven chunks - in this document, 60 to 1,800 characters.
#
# PARAGRAPH_MAX is the safety net: a paragraph longer than this is still split
# recursively, because a single 5,000-character chunk would dilute its own
# embedding until it matched nothing in particular.
PARAGRAPH_MAX = 1500

BAR = "=" * 74


# --- helpers --------------------------------------------------------------

def make_splitter(chunk_size: int = CHUNK_SIZE,
                  overlap: int = CHUNK_OVERLAP) -> RecursiveCharacterTextSplitter:
    """The one splitter definition both tools use."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
        length_function=len,
    )


def split_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split a raw string into chunks."""
    return make_splitter(chunk_size, overlap).split_text(text)


def split_documents(docs: list[Document], chunk_size: int = CHUNK_SIZE,
                    overlap: int = CHUNK_OVERLAP) -> list[Document]:
    """Split Documents, keeping metadata and numbering each chunk."""
    chunks = make_splitter(chunk_size, overlap).split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk"] = i
    return chunks


def split_documents_by_paragraph(docs: list[Document],
                                 max_chars: int = PARAGRAPH_MAX
                                 ) -> list[Document]:
    """One chunk per paragraph, keeping each idea intact.

    Written by hand rather than with RecursiveCharacterTextSplitter, because
    that splitter *packs*: given a character budget it merges consecutive
    small pieces until they fill it. That is the opposite of what is wanted
    here - two unrelated short paragraphs would end up sharing one vector.

    A paragraph over max_chars is still split recursively, since an
    over-long chunk averages too many ideas into one vector to match well.
    """
    splitter = make_splitter(max_chars, CHUNK_OVERLAP)
    chunks: list[Document] = []

    for doc in docs:
        # Blank-line-separated first; falls back to single newlines, which is
        # what these documents actually use.
        blocks = [p.strip() for p in doc.page_content.split("\n\n")]
        if len(blocks) <= 1:
            blocks = [p.strip() for p in doc.page_content.split("\n")]
        paragraphs = [p for p in blocks if p]

        for para_no, para in enumerate(paragraphs):
            pieces = [para] if len(para) <= max_chars else splitter.split_text(para)
            for part_no, piece in enumerate(pieces):
                meta = dict(doc.metadata)
                meta["paragraph"] = para_no
                # Only set when a paragraph had to be broken up, so it is
                # obvious in the output which chunks are partial.
                if len(pieces) > 1:
                    meta["part"] = f"{part_no + 1}/{len(pieces)}"
                chunks.append(Document(page_content=piece, metadata=meta))

    for i, c in enumerate(chunks):
        c.metadata["chunk"] = i
    return chunks


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """Every .txt and .md under docs/, so new files are picked up on rebuild."""
    docs: list[Document] = []
    for pattern in ("**/*.txt", "**/*.md"):
        loader = DirectoryLoader(
            str(docs_dir),
            glob=pattern,
            # docs/img/ holds screenshots and their caption file - assets, not
            # source material. Indexing them would put noise in retrieval.
            exclude=["**/img/**"],
            loader_cls=TextLoader,
            # Explicit encoding: the default on Windows is cp1252, which fails
            # on the curly quotes in these documents.
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        )
        docs.extend(loader.load())

    for d in docs:
        d.metadata["file"] = Path(d.metadata.get("source", "?")).name
    return docs


def embeddings(model: str = EMBED_MODEL) -> OpenAIEmbeddings:
    """The embedding model.

    Indexing and querying must use the same one - vectors from different
    models are not comparable, and mixing them degrades results silently
    rather than raising an error.
    """
    return OpenAIEmbeddings(model=model)


def count_tokens(text: str) -> int:
    """Exact token count for cost estimates; falls back to a rough ratio."""
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return len(text) // 4


def estimate_cost(texts: list[str]) -> tuple[int, float]:
    tokens = sum(count_tokens(t) for t in texts)
    return tokens, tokens / 1_000_000 * EMBED_PRICE_PER_1M_TOKENS


def preview(text: str, width: int = 64) -> str:
    """One-line snippet of a chunk, whitespace collapsed."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def chunk_stats(chunks: list[str]) -> str:
    sizes = [len(c) for c in chunks]
    return (f"{len(chunks)} (smallest {min(sizes)}, largest {max(sizes)}, "
            f"average {sum(sizes) // len(sizes)})")
