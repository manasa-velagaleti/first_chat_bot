# Embeddings CLI

Turns text into OpenAI embeddings and shows you what comes back — the chunks, the
vectors, and how they were split.

Separate from the chat assistant in the parent folder; it only shares the `.env`
file and the `docs/` source text.

## Run it

```bash
# from the project root
.venv\Scripts\python.exe embeddings_app/embed.py
```

With no arguments it embeds `docs/sotu_address_obama.txt`.

```bash
--text "some string"       embed a string instead of a file
--file path/to/file.txt    embed a different file
--chunk-size 500           max characters per chunk (default 500)
--overlap 50               characters shared between neighbours (default 50)
--limit 5                  only the first N chunks - keeps cost down
--full                     print all 1536 numbers instead of the first 8
--save out.json            write chunks + vectors to JSON
--dry-run                  chunk and cost it without calling the API
```

**Start with `--dry-run`.** It shows exactly how your text will be split and what the
call would cost, without spending anything.

## What the output means

```
[2] 482 chars | 93 tokens
     text : "It's tempting to look back on these moments and assume that our…"
     dims : 1536
     vec  : [+0.01271, -0.02696, +0.03922, ... ]   (1528 more)
     norm : 0.999764
```

| Field | Meaning |
|-------|---------|
| `[2]` | Chunk index, in document order |
| `chars / tokens` | Size. Tokens are what you're billed for |
| `text` | The chunk this vector represents |
| `dims` | 1536 for `text-embedding-3-small` |
| `vec` | First 8 of 1536 numbers. Individually meaningless — only *distances between whole vectors* carry meaning |
| `norm` | Vector length. Always ≈1.0, because OpenAI returns normalised vectors. A different value would mean something rescaled them |

## Recursive chunking

The whole document is one long string, but embedding it in one piece would produce a
single vector averaging everything — useless for finding specific passages. So it gets
split first.

`RecursiveCharacterTextSplitter` tries separators **in order** and only falls to the
next when a piece is still too large:

```
"\n\n"  paragraph break   ← try hardest to split here
"\n"    line break
". "    sentence end
" "     word gap
""      raw character      ← last resort
```

That ordering is what "meaningful breaking" means. A paragraph under 500 characters
stays whole. A longer one splits at sentence ends rather than mid-word. Only genuinely
unbreakable text gets cut arbitrarily.

On the SOTU address at size 500:

```
122 chunks, smallest 45 chars, largest 498, average 335
```

Nothing hits the 500 ceiling exactly — every chunk ended at a natural boundary before
it had to.

**Overlap** (default 50 chars) repeats the tail of each chunk at the head of the next,
so a sentence spanning a boundary still appears intact in one of them.

## Cost

`text-embedding-3-small` is **$0.02 per million tokens**. The entire 41,000-character
address is 8,299 tokens — about **$0.0002**. The CLI prints an estimate before every
call.

## Saved JSON

`--save` writes:

```jsonc
{
  "model": "text-embedding-3-small",
  "source": "docs/sotu_address_obama.txt",
  "chunk_size": 500,
  "chunk_overlap": 50,
  "dimensions": 1536,
  "chunks": [{"index": 0, "text": "...", "embedding": [0.0258, ...]}]
}
```

It's large — the full address comes to **5.5 MB**, since 122 × 1536 floats is a lot of
text. `embeddings_app/out/` is gitignored for that reason.

## Where this goes next

Embeddings become useful when you *compare* them. Cosine similarity between two
vectors measures how related their texts are, which is the basis of semantic search
and RAG: embed a question, find the nearest chunks, hand those to a chat model as
context. The saved JSON is enough to build that on top of.
