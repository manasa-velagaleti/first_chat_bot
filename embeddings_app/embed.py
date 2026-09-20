"""Turn text into OpenAI embeddings and show what comes back.

    python embeddings_app/embed.py                      # the Obama SOTU address
    python embeddings_app/embed.py --text "hello there"
    python embeddings_app/embed.py --file notes.txt --limit 5

Chunks the input with RecursiveCharacterTextSplitter, embeds each chunk with
text-embedding-3-small, and prints the vectors alongside the text they came
from.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# This app lives in a subfolder, but the API key lives in the project root.
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DEFAULT_FILE = ROOT / "docs" / "sotu_address_obama.txt"
MODEL = "text-embedding-3-small"
PRICE_PER_1M_TOKENS = 0.02          # USD, text-embedding-3-small

BAR = "=" * 74


def count_tokens(text: str) -> int:
    """Token count for cost estimation. Falls back to a rough ratio."""
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return len(text) // 4


def split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Break text on meaningful boundaries.

    RecursiveCharacterTextSplitter tries separators in order - paragraph
    break, then line break, then space, then raw character - and only falls
    to the next one when a piece is still too big. So a paragraph that fits
    stays whole instead of being cut mid-word.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


def preview(text: str, width: int = 64) -> str:
    """One-line, quote-safe snippet of a chunk."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def show(index: int, chunk: str, vector: list[float], show_full: bool) -> None:
    norm = math.sqrt(sum(v * v for v in vector))
    head = ", ".join(f"{v:+.5f}" for v in vector[:8])

    print(f"\n[{index}] {len(chunk)} chars | {count_tokens(chunk)} tokens")
    print(f'     text : "{preview(chunk)}"')
    print(f"     dims : {len(vector)}")
    if show_full:
        print(f"     vec  : {vector}")
    else:
        print(f"     vec  : [{head}, ... ]   ({len(vector) - 8} more)")
    # OpenAI returns unit vectors, so this should read 1.000000. A different
    # value means something re-scaled the numbers on the way through.
    print(f"     norm : {norm:.6f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--text", help="embed this string instead of a file")
    src.add_argument("--file", type=Path, help="embed this file")
    ap.add_argument("--chunk-size", type=int, default=500,
                    help="max characters per chunk (default: 500)")
    ap.add_argument("--overlap", type=int, default=50,
                    help="characters shared between neighbours (default: 50)")
    ap.add_argument("--limit", type=int,
                    help="only embed the first N chunks - keeps cost down "
                         "while experimenting")
    ap.add_argument("--full", action="store_true",
                    help="print every number instead of the first 8")
    ap.add_argument("--save", type=Path,
                    help="also write chunks + vectors to this JSON file")
    ap.add_argument("--dry-run", action="store_true",
                    help="chunk and cost it, but make no API call")
    args = ap.parse_args()

    # --- load the text ----------------------------------------------------
    if args.text:
        text, source = args.text, "--text argument"
    else:
        path = args.file or DEFAULT_FILE
        if not path.exists():
            print(f"File not found: {path}")
            return 1
        text = path.read_text(encoding="utf-8")
        source = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)

    # --- chunk ------------------------------------------------------------
    chunks = split(text, args.chunk_size, args.overlap)
    total_chunks = len(chunks)
    if args.limit:
        chunks = chunks[: args.limit]
    if not chunks:
        print("Nothing to embed - the input was empty.")
        return 1

    tokens = sum(count_tokens(c) for c in chunks)
    sizes = [len(c) for c in chunks]

    print(BAR)
    print(f"Source     : {source}")
    print(f"Input      : {len(text):,} chars")
    print(f"Chunking   : recursive, size {args.chunk_size}, overlap {args.overlap}")
    limited = f" of {total_chunks} (--limit)" if len(chunks) < total_chunks else ""
    print(f"Chunks     : {len(chunks)}{limited}  "
          f"(smallest {min(sizes)}, largest {max(sizes)}, "
          f"average {sum(sizes) // len(sizes)} chars)")
    print(f"Model      : {MODEL}")
    print(f"Tokens     : {tokens:,}  "
          f"(about ${tokens / 1_000_000 * PRICE_PER_1M_TOKENS:.6f})")
    print(BAR)

    if args.dry_run:
        for i, chunk in enumerate(chunks):
            print(f"\n[{i}] {len(chunk)} chars")
            print(f'     text : "{preview(chunk)}"')
        print(f"\nDry run - no API call made. Drop --dry-run to embed.")
        return 0

    # --- embed ------------------------------------------------------------
    # One call for the whole list: batching is far faster than a call per
    # chunk, and the vectors come back in the order they were sent.
    try:
        embedder = OpenAIEmbeddings(model=MODEL)
        vectors = embedder.embed_documents(chunks)
    except Exception as err:
        print(f"\nEmbedding failed: {type(err).__name__}: {err}")
        if "api_key" in str(err).lower() or "401" in str(err):
            print("Check OPENAI_API_KEY in your .env file.")
        return 1

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        show(i, chunk, vector, args.full)

    print(f"\n{BAR}")
    print(f"Embedded {len(vectors)} chunks into {len(vectors[0])}-dimensional vectors.")
    print(BAR)

    if args.save:
        payload = {
            "model": MODEL,
            "source": source,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.overlap,
            "dimensions": len(vectors[0]),
            "chunks": [
                {"index": i, "text": c, "embedding": v}
                for i, (c, v) in enumerate(zip(chunks, vectors))
            ],
        }
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        size_mb = args.save.stat().st_size / 1_000_000
        print(f"Saved to {args.save}  ({size_mb:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
