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
import logging
from pathlib import Path

from dotenv import load_dotenv

warnings.filterwarnings("ignore")   # langchain-community sunset notice

from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
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

# And the floor. A PDF page is full of one- and two-word lines - headings,
# table cells, page numbers - and splitting on newlines turns each into its
# own "paragraph". Embedding a 1-character chunk produces a vector that means
# nothing and can still win a similarity search, so anything shorter than this
# is dropped. Plain prose is unaffected; the shortest real paragraph in the
# SOTU address is 20 characters, which is why this sits below that.
PARAGRAPH_MIN = 15

BAR = "=" * 74

# --- supported file types -------------------------------------------------
# extension -> (loader class, extra kwargs). Adding a format is one line here;
# load_documents() and the upload endpoint both read from this, so neither can
# drift from the other about what is accepted.
#
# utf-8 is explicit on the text loaders: the default on Windows is cp1252,
# which fails on curly quotes and em-dashes. Binary formats carry their own
# encoding, so they take no such argument.
LOADERS: dict[str, tuple[type, dict]] = {
    ".txt":  (TextLoader, {"encoding": "utf-8"}),
    ".md":   (TextLoader, {"encoding": "utf-8"}),
    ".pdf":  (PyPDFLoader, {}),          # one Document per page
    ".docx": (Docx2txtLoader, {}),
    ".csv":  (CSVLoader, {"encoding": "utf-8"}),   # one Document per row
}

SUPPORTED = tuple(LOADERS)


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
        paragraphs = [p for p in blocks if len(p) >= PARAGRAPH_MIN]

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

    # docs/img/ holds screenshots and their caption file - assets, not source
    # material. Indexing them would put noise in retrieval.
    exclude = ["**/img/**"]

    for ext, (loader_cls, kwargs) in LOADERS.items():
        if ext == ".pdf":
            # One at a time, so a scanned PDF can fall back to OCR on its own
            # without the others paying for it.
            for pdf in sorted(docs_dir.rglob("*.pdf")):
                if "img" in pdf.parts:
                    continue
                try:
                    docs.extend(load_pdf(pdf))
                except Exception:
                    continue        # unreadable file: skip, don't abort
            continue

        docs.extend(DirectoryLoader(
            str(docs_dir),
            glob=f"**/*{ext}",
            exclude=exclude,
            loader_cls=loader_cls,
            loader_kwargs=kwargs,
            silent_errors=True,
        ).load())

    return _tidy(docs)


def load_pdf(path: Path) -> list[Document]:
    """Read a PDF, falling back to OCR when it has no text layer.

    A PDF made by a word processor carries real text and reads in a moment.
    A scanned one is photographs of paper: pypdf finds nothing in it, which
    is not an error - there genuinely is no text, only pixels.

    So: try the fast path, and only when it comes back empty spend the time
    on OCR. Running OCR unconditionally would add minutes to every ordinary
    PDF for no benefit.
    """
    docs = PyPDFLoader(str(path)).load()
    if sum(len(d.page_content.strip()) for d in docs):
        return docs

    ocr = ocr_pdf(path)
    return ocr if ocr else docs


def ocr_pdf(path: Path, scale: float = 2.0) -> list[Document]:
    """OCR a PDF by rendering each page to a bitmap and reading that.

    Rendering rather than pulling out the embedded images: a scanner can
    store a page as CCITT or JBIG2, or as dozens of image strips, and
    extracting those relies on the PDF library decoding each format. Drawing
    the page the way a viewer would sidesteps all of it - whatever the page
    looks like on screen is what gets read.

    scale 2.0 is roughly 144 dpi. Higher reads small print better and costs
    proportionally more time.
    """
    try:
        import numpy as np
        import pypdfium2 as pdfium
        from rapidocr import RapidOCR
    except ImportError:
        return []          # OCR extras not installed

    # RapidOCR logs several lines about model files on every single call.
    noisy = logging.getLogger("RapidOCR")
    was = noisy.level
    noisy.setLevel(logging.ERROR)
    try:
        engine = RapidOCR()
        pdf = pdfium.PdfDocument(str(path))
        out: list[Document] = []
        for i in range(len(pdf)):
            image = pdf[i].render(scale=scale).to_pil().convert("RGB")
            result = engine(np.array(image))
            text = "\n".join(result.txts) if result and result.txts else ""
            if text.strip():
                out.append(Document(
                    page_content=text,
                    metadata={"source": str(path), "page": i, "ocr": True},
                ))
        return out
    except Exception:
        return []
    finally:
        noisy.setLevel(was)


def describe_pdf(path: Path) -> str:
    """Why a PDF yielded no text - for an error message worth acting on."""
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(path))
        return f"{len(pdf)} page(s), no text layer and OCR found nothing readable"
    except Exception as exc:
        msg = str(exc).lower()
        if "password" in msg or "encrypt" in msg:
            return "the file is password-protected"
        return f"the file could not be opened ({type(exc).__name__})"


def load_one(path: Path) -> list[Document]:
    """Load a single file.

    Used to validate an upload. Checking one file this way takes a moment;
    re-reading the whole corpus to check it took ten seconds once a large PDF
    was present, which looked to the user like a failed upload.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _tidy(load_pdf(path))

    loader_cls, kwargs = LOADERS.get(suffix, (None, None))
    if loader_cls is None:
        raise ValueError(f"{path.suffix} is not a supported file type")
    return _tidy(loader_cls(str(path), **(kwargs or {})).load())


def _tidy(docs: list[Document]) -> list[Document]:
    for d in docs:
        d.metadata["file"] = Path(d.metadata.get("source", "?")).name
        # PyPDFLoader numbers pages from 0; +1 matches what a reader shows.
        if "page" in d.metadata:
            try:
                d.metadata["page"] = int(d.metadata["page"]) + 1
            except (TypeError, ValueError):
                pass
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
