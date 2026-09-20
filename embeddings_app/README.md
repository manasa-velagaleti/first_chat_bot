# Embeddings & RAG

Three CLIs over one pipeline. `embed.py` shows you what embeddings *are*;
`rag.py` and `rag_pinecone.py` use them to answer questions about the documents
in `docs/`.

**Stack:** LangChain · FAISS / Pinecone · OpenAI `text-embedding-3-small` + `gpt-4.1-mini`

```
core.py     load docs -> chunk -> embedding model
              │
              ├── embed.py          fixed 500  -> print the vectors
              │
              ├── rag.py            fixed 500  -> FAISS (local disk) -> answer
              │
              └── rag_pinecone.py   paragraph  -> Pinecone (cloud)   -> answer
```

### The two RAG apps, side by side

|  | `rag.py` | `rag_pinecone.py` |
|---|---|---|
| Vector store | FAISS, a folder on disk | Pinecone, hosted |
| Chunking | fixed 500 chars, recursive on `\n` | **one chunk per paragraph** |
| Chunks | 122 (45–498 chars) | 107 (20–1,172 chars) |
| Score | L2 distance — **lower** is better | cosine similarity — **higher** is better |
| Needs | nothing beyond OpenAI | `PINECONE_API_KEY` |
| Survives | a laptop | a laptop dying |

Keeping both is the point: same documents, same embedding model, two chunking
strategies and two databases. Ask the same question of each and the differences
are visible in what comes back.

`core.py` holds everything the two have in common. That isn't tidiness for its own
sake: if chunking drifted between them, a chunk you inspected with `embed.py` would
not be the chunk `rag.py` actually retrieved, and the inspection tool would be lying
about the system it exists to explain. They now provably produce identical chunks.

## Files

| File | Role |
|------|------|
| `core.py` | Shared: paths, models, both chunking strategies, document loading, cost |
| `embed.py` | Inspect embeddings — chunk, embed, print vectors |
| `rag.py` | Ask questions — FAISS on disk, fixed-size chunks |
| `rag_pinecone.py` | Ask questions — Pinecone cloud, paragraph chunks |

---

## `rag_pinecone.py` — Pinecone + paragraph chunking

```bash
.venv\Scripts\python.exe embeddings_app/rag_pinecone.py --build   # create + upload
.venv\Scripts\python.exe embeddings_app/rag_pinecone.py           # interactive
.venv\Scripts\python.exe embeddings_app/rag_pinecone.py --ask "What about jobs?"
.venv\Scripts\python.exe embeddings_app/rag_pinecone.py --stats   # what's in there
```

```
--build              create the index and upload docs/, then exit
--ask "question"     ask once and exit
--stats              show dimension, host and vector counts
--index NAME         a different index (default: sotu-paragraphs)
--namespace NAME     keep several document sets in one index
-k 4                 how many chunks to retrieve
--model gpt-4.1      a different OpenAI chat model
--no-sources         hide the retrieved excerpts
```

Add `PINECONE_API_KEY` to `.env` first — free tier at
[app.pinecone.io](https://app.pinecone.io) covers this comfortably.

`--build` creates a serverless index (1536 dims, cosine, `aws/us-east-1`), waits
for it to become ready, **clears any previous vectors**, then uploads. Without
that clear, a second `--build` would leave two copies of every chunk in the index.

```
Retrieved 3 excerpts (cosine similarity: higher = closer match)
  [1] score 0.564 | sotu_address_obama.txt paragraph 33
      So tonight, I'm proposing that we take $30 billion of the money Wall…
  [2] score 0.505 | sotu_address_obama.txt paragraph 25
      Talk to the small business in Phoenix that will triple its workforce…
```

Sources cite the **paragraph number**, so a retrieved excerpt maps to a place in
the original document.

### Paragraph chunking

One chunk per paragraph, rather than a fixed character budget. Each vector then
represents one complete idea instead of an arbitrary 500-character window.

It's written by hand in `core.split_documents_by_paragraph()` rather than with
`RecursiveCharacterTextSplitter`, because that splitter **packs**: given a
budget it merges consecutive small pieces until they fill it. That's the exact
opposite of what's wanted here — two unrelated short paragraphs would end up
sharing a vector.

`PARAGRAPH_MAX` (1500) is the safety net. A paragraph longer than that is split
recursively anyway, because one enormous chunk averages too many ideas into a
single vector to match anything well. On this document nothing hit it — the
longest paragraph is 1,172 characters.

The trade-off is even chunk sizes versus whole ideas:

```
fixed-500   : 122 chunks,   45 - 498 chars,  average 335
paragraph   : 107 chunks,   20 - 1,172 chars, average 381
```

> **Pinecone scores are cosine similarity — higher is better.** FAISS in `rag.py`
> returns L2 distance, where lower is better. Same pipeline, opposite direction;
> easy to misread when comparing the two.

---

## `rag.py` — question answering

```bash
.venv\Scripts\python.exe embeddings_app/rag.py --build   # index docs/ first
.venv\Scripts\python.exe embeddings_app/rag.py           # interactive
.venv\Scripts\python.exe embeddings_app/rag.py --ask "What about jobs?"
```

```
--build              (re)build the index from docs/, then exit
--ask "question"     ask once and exit
-k 4                 how many chunks to retrieve (default 4)
--model gpt-4.1      a different OpenAI chat model
--no-sources         hide the retrieved excerpts
```

Re-run `--build` after adding or editing anything in `docs/`. The index is a
snapshot and won't notice changes on its own.

### What it looks like

```
> What did Obama propose to help small businesses?

Obama proposed several measures:
- Using $30 billion repaid by Wall Street banks to help community banks give
  small businesses the credit they need to stay afloat [1].
- A new small business tax credit for firms that hire or raise wages [1].
- Eliminating all capital gains taxes on small business investment [2].

--------------------------------------------------------------------------
Retrieved 4 excerpts (distance: lower = closer match)
  [1] distance 0.810 | sotu_address_obama.txt chunk 25
      So tonight, I'm proposing that we take $30 billion of the money Wall…
  [2] distance 1.028 | sotu_address_obama.txt chunk 26
      . While we're at it, let's also eliminate all capital gains taxes on…
```

Citations map to the excerpts below the answer, so every claim traces to a passage
you can read.

---

## `embed.py` — seeing the vectors

```bash
.venv\Scripts\python.exe embeddings_app/embed.py --dry-run   # free: just chunking
.venv\Scripts\python.exe embeddings_app/embed.py --limit 3
.venv\Scripts\python.exe embeddings_app/embed.py --text "hello there"
```

```
--text / --file      what to embed (default: the SOTU address)
--chunk-size 500     experiment with chunking
--overlap 50
--limit N            only the first N chunks
--full               all 1536 numbers instead of the first 8
--save out.json      write chunks + vectors to a file
--dry-run            chunk and cost it without calling the API
```

```
[2] 482 chars | 93 tokens
     text : "It's tempting to look back on these moments and assume that our…"
     dims : 1536
     vec  : [+0.01271, -0.02696, +0.03922, ... ]   (1528 more)
     norm : 0.999764
```

`norm` is a free sanity check — OpenAI returns unit vectors, so it always reads
≈1.0. Anything else would mean something rescaled the numbers.

**Use `--dry-run` when tuning `--chunk-size`.** It shows exactly how the text will
split and what the call would cost, without spending anything.

---

## How the pipeline works

### 1. Load

`DirectoryLoader` reads every `.txt` and `.md` in `docs/`, so dropping a new
document in and re-running `--build` picks it up. `docs/img/` is excluded — those
are screenshots and captions, and indexing them would put noise in retrieval.

### 2. Split — 500 chars, recursive, on `\n`

```python
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
```

Tried **in order**, dropping to the next only when a piece is still over 500
characters.

`"\n"` is what does the work here. The address puts **one paragraph per line with no
blank lines** — 106 single newlines and **zero** `\n\n`. LangChain's default list
leads with `\n\n`, which would never match, so the split would fall straight through
to sentence boundaries and cut paragraphs apart. Naming `\n` explicitly keeps them
intact.

Result: **122 chunks, largest 498 characters.** Nothing hit the 500 ceiling — every
chunk found a natural boundary first.

### 3. Embed

Each chunk becomes a 1536-dimensional vector. Similar meanings land close together,
which is why *"How many people are out of work?"* retrieves *"One in ten Americans
still cannot find work"* despite sharing no keywords. Keyword search would miss it.

### 4. Store and retrieve (rag.py)

FAISS holds the vectors and searches by nearest neighbour, saved to
`embeddings_app/faiss_index/` so you only pay to embed once. Your question is
embedded **with the same model** — vectors from different models aren't comparable,
and mixing them degrades results silently rather than erroring.

> **The score is L2 distance, not similarity. Lower is closer.** A relevant chunk
> scores ~0.8, an unrelated one ~1.9. Easy to misread as a 0–1 confidence.

### 5. Generate (rag.py)

Excerpts plus question go to `gpt-4.1-mini` at `temperature=0`, with a system prompt
that says: use only these excerpts, cite them by number, say so if they don't answer
the question.

That last rule is what stops invention. Asked *"What is Obama's favourite pizza
topping?"* it replies: *"The excerpts do not provide any information about Obama's
favourite pizza topping."*

---

## Cost

| | |
|---|---|
| Indexing (one-off) | 8,299 tokens ≈ **$0.0002** |
| Each question | ~2,000 in + ~200 out ≈ **$0.0005** |

`embed.py --dry-run` prices any chunking change before you pay for it.

## Notes

**`allow_dangerous_deserialization=True`** is required to load the FAISS index —
its metadata is pickle-backed, so loading executes code. Safe for an index this
script wrote on your machine; never point it at a downloaded one.

**`faiss_index/` and `out/` are gitignored.** Both are build artefacts — regenerate
with `--build`.

**`langchain-community` prints a sunset warning.** FAISS still lives there; the
standalone `langchain-faiss` package on PyPI is an empty placeholder that exports
nothing. `core.py` suppresses the warning, which is why `rag.py` imports `core`
*before* `langchain_community` — the warning fires at import time.

## Tuning

| Symptom | Try |
|---------|-----|
| Answers miss context that's clearly in the document | raise `-k` to 6–8 |
| Answers pull in irrelevant material | lower `-k` to 2–3 |
| Chunks cut mid-thought | raise `CHUNK_SIZE`, or add a separator matching the document's structure |
| Answers feel thin | `--model gpt-4.1` |

`CHUNK_SIZE` and `SEPARATORS` live in `core.py` and apply to both tools. Chunking
only takes effect at index time, so re-run `--build` after changing them.
