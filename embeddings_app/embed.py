"""Inspect embeddings: chunk text, embed it, and show the vectors.

    python embeddings_app/embed.py                      # the Obama SOTU address
    python embeddings_app/embed.py --text "hello there"
    python embeddings_app/embed.py --file notes.txt --limit 5

This is the "look at what's happening" tool. It shares chunking and the
embedding model with rag.py through core.py, so the chunks you inspect here
are exactly the chunks rag.py retrieves.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import core

DEFAULT_FILE = core.DOCS_DIR / "sotu_address_obama.txt"


def show(index: int, chunk: str, vector: list[float], show_full: bool) -> None:
    norm = math.sqrt(sum(v * v for v in vector))
    head = ", ".join(f"{v:+.5f}" for v in vector[:8])

    print(f"\n[{index}] {len(chunk)} chars | {core.count_tokens(chunk)} tokens")
    print(f'     text : "{core.preview(chunk)}"')
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
    ap.add_argument("--chunk-size", type=int, default=core.CHUNK_SIZE,
                    help=f"max characters per chunk (default: {core.CHUNK_SIZE})")
    ap.add_argument("--overlap", type=int, default=core.CHUNK_OVERLAP,
                    help=f"characters shared between neighbours "
                         f"(default: {core.CHUNK_OVERLAP})")
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
        source = str(path.relative_to(core.ROOT)
                     if path.is_relative_to(core.ROOT) else path)

    # --- chunk ------------------------------------------------------------
    chunks = core.split_text(text, args.chunk_size, args.overlap)
    total_chunks = len(chunks)
    if args.limit:
        chunks = chunks[: args.limit]
    if not chunks:
        print("Nothing to embed - the input was empty.")
        return 1

    tokens, cost = core.estimate_cost(chunks)
    limited = f" of {total_chunks} (--limit)" if len(chunks) < total_chunks else ""

    print(core.BAR)
    print(f"Source     : {source}")
    print(f"Input      : {len(text):,} chars")
    print(f"Chunking   : recursive, size {args.chunk_size}, "
          f"overlap {args.overlap}")
    print(f"Separators : {core.SEPARATORS}")
    print(f"Chunks     : {core.chunk_stats(chunks)}{limited}")
    print(f"Model      : {core.EMBED_MODEL}")
    print(f"Tokens     : {tokens:,}  (about ${cost:.6f})")
    print(core.BAR)

    if args.dry_run:
        for i, chunk in enumerate(chunks):
            print(f"\n[{i}] {len(chunk)} chars")
            print(f'     text : "{core.preview(chunk)}"')
        print("\nDry run - no API call made. Drop --dry-run to embed.")
        return 0

    # --- embed ------------------------------------------------------------
    # One call for the whole list: batching is far faster than a call per
    # chunk, and the vectors come back in the order they were sent.
    try:
        vectors = core.embeddings().embed_documents(chunks)
    except Exception as err:
        print(f"\nEmbedding failed: {type(err).__name__}: {err}")
        if "api_key" in str(err).lower() or "401" in str(err):
            print("Check OPENAI_API_KEY in your .env file.")
        return 1

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        show(i, chunk, vector, args.full)

    print(f"\n{core.BAR}")
    print(f"Embedded {len(vectors)} chunks into "
          f"{len(vectors[0])}-dimensional vectors.")
    print(core.BAR)

    if args.save:
        payload = {
            "model": core.EMBED_MODEL,
            "source": source,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.overlap,
            "separators": core.SEPARATORS,
            "dimensions": len(vectors[0]),
            "chunks": [
                {"index": i, "text": c, "embedding": v}
                for i, (c, v) in enumerate(zip(chunks, vectors))
            ],
        }
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved to {args.save}  "
              f"({args.save.stat().st_size / 1_000_000:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
