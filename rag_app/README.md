# RAG over `docs/`

Ask questions about the documents in the project's `docs/` folder and get answers
grounded in them — with the exact passages used shown underneath.

**Stack:** LangChain · FAISS · OpenAI `text-embedding-3-small` + `gpt-4.1-mini`

## Run it

```bash
# from the project root
.venv\Scripts\python.exe rag_app/rag.py --build    # index the documents first
.venv\Scripts\python.exe rag_app/rag.py            # interactive Q&A
```

```bash
--build                (re)build the index from docs/, then exit
--ask "question"       ask once and exit
-k 4                   how many chunks to retrieve (default 4)
--model gpt-4.1        a different OpenAI chat model
--no-sources           hide the retrieved excerpts
```

Re-run `--build` whenever you add or change a file in `docs/`. The index is a
snapshot; it doesn't notice edits on its own.

## What it looks like

```
> What did Obama propose to help small businesses?

Obama proposed several measures to help small businesses:
- Using $30 billion repaid by Wall Street banks to help community banks give
  small businesses the credit they need to stay afloat [1].
- A new small business tax credit for over one million small businesses that
  hire new workers or raise wages [1].
- Eliminating all capital gains taxes on small business investment [2].

--------------------------------------------------------------------------
Retrieved 4 excerpts (distance: lower = closer match)
  [1] distance 0.810 | sotu_address_obama.txt chunk 25
      So tonight, I'm proposing that we take $30 billion of the money Wall…
  [2] distance 1.028 | sotu_address_obama.txt chunk 26
      . While we're at it, let's also eliminate all capital gains taxes on…
```

The citations `[1]` `[2]` map to the excerpts listed below the answer, so every
claim is traceable to a passage you can read.

## The pipeline

```
          ── INDEX (once, or after documents change) ──
docs/*.txt  →  split into 500-char chunks  →  embed each  →  FAISS index on disk
                  (recursive, on "\n")         (1536 dims)      (122 vectors)

          ── ASK (every question) ──
question  →  embed  →  nearest 4 chunks  →  chunks + question  →  answer
              ↑                                                    + citations
              same embedding model as the index
```

### 1. Load

`DirectoryLoader` reads every `.txt` and `.md` in `docs/`, so dropping another
document in and re-running `--build` picks it up. `docs/img/` is excluded — those
are screenshots and captions, not source material, and indexing them would put
noise into retrieval.

### 2. Split — 500 chars, recursive, on `\n`

```python
separators = ["\n\n", "\n", ". ", " ", ""]
```

The splitter tries these **in order**, dropping to the next only when a piece is
still over 500 characters.

`"\n"` is what does the work here. This document puts **one paragraph per line with
no blank lines** — it contains 106 single newlines and **zero** `\n\n`. The library's
default separator list leads with `\n\n`, which would never match, so the split
would fall straight through to sentence and word boundaries. Naming `\n` explicitly
is what keeps paragraphs intact.

Result: 122 chunks, largest **498** characters. Nothing hit the 500 ceiling — every
chunk found a natural boundary first.

### 3. Embed and store

Each chunk becomes a 1536-dimensional vector via `text-embedding-3-small`. FAISS
stores them and searches by nearest neighbour. The index is written to
`rag_app/faiss_index/` so you only pay to embed once.

### 4. Retrieve

Your question is embedded **with the same model** and FAISS returns the `k` closest
chunks. Same model matters: vectors from different models aren't comparable, and
mixing them gives silently useless results rather than an error.

This is why RAG beats keyword search — *"How many people are out of work?"* retrieves
*"One in ten Americans still cannot find work"* despite sharing almost no words.

> **The score is L2 distance, not similarity.** **Lower is closer.** A relevant chunk
> scores ~0.8; an unrelated one ~1.9. It is easy to misread as a 0-1 confidence.

### 5. Generate

The excerpts and the question go to `gpt-4.1-mini` at `temperature=0` with a system
prompt that says: use only these excerpts, cite them by number, and say so if they
don't answer the question.

That last instruction is what stops it inventing things. Asked *"What is Obama's
favourite pizza topping?"* it answers: *"The excerpts do not provide any information
about Obama's favourite pizza topping."*

## Cost

| | |
|---|---|
| Indexing (one-off) | 8,299 tokens ≈ **$0.0002** |
| Each question | ~2,000 input + ~200 output tokens ≈ **$0.0005** |

Indexing is paid once; re-running `--build` pays again.

## Notes

**`allow_dangerous_deserialization=True`** is needed to load the index. FAISS stores
its metadata with pickle, so loading executes code. Safe here because the index was
written by this script on your machine — never point it at an index downloaded from
elsewhere.

**`faiss_index/` is gitignored.** It's a build artefact; regenerate it with `--build`.

**`langchain-community` prints a sunset warning.** FAISS still lives there — the
standalone `langchain-faiss` package on PyPI is an empty placeholder that exports
nothing. The warning is suppressed at the top of `rag.py`.

## Tuning

| Symptom | Try |
|---------|-----|
| Answers miss context that's clearly in the document | raise `-k` to 6-8 |
| Answers pull in irrelevant material | lower `-k` to 2-3 |
| Chunks cut mid-thought | raise `CHUNK_SIZE`, or add a separator that matches the document's structure |
| Answers feel thin | try `--model gpt-4.1` |

Change `CHUNK_SIZE` or `SEPARATORS` in `rag.py` and re-run `--build` — chunking only
takes effect at index time.
