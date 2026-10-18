# Demo script

A ten-minute walkthrough. Each step says what to click, what to say, and what
should happen.

---

## Before you start (5 minutes ahead)

```powershell
cd C:\Users\prati\Desktop\GEN-AI\prompt_llm_sdk_framework
.venv\Scripts\python.exe -m streamlit run app.py
```

Then, in the sidebar:

| Setting | Use | Why |
|---------|-----|-----|
| **Provider** | **OpenAI** | Most reliable. Gemini free tier runs out mid-demo |
| **Model** | `gpt-4.1-mini` | Fast, cheap, calls tools reliably |
| Judge provider | **Groq** | Must differ from the answering model |
| Judge model | `openai/gpt-oss-120b` | Fast, free tier |

**Warm it up.** Ask one throwaway question before the audience is watching —
the first call to each service is slower while connections open.

**Avoid `gemini-3.7-flash`.** It is quota-exhausted and will error. If you
want to show Gemini, use **`gemini-3.5-flash`**.

**Check the mic** if you plan to demo voice. Click 🎤, allow the browser
prompt, say something, confirm text appears.

---

## The documents, and what each one is good for

Five are already indexed — 587 chunks, 1536 dimensions.

| Document | Use it to show |
|----------|----------------|
| **`qwerty1.pdf`** — Siemens job posting | **The main demo.** A scanned PDF read by OCR. Specific facts to check: salary, experience, languages |
| **`_10-K-2025-As-Filed.pdf`** — 80 pages | Scale. 466 of the 587 chunks |
| **`sotu_address_obama.txt`** | Switching documents mid-conversation |
| **`Introduction_to_AI_Agents.pdf`** | **The honest moment** — it is a *certificate*, 1 chunk. Good for "the filename is not the content" |
| `Hackathon_Problem_Statement...` | Spare |

**Lead with `qwerty1.pdf`.** It is a real document, it was unreadable until OCR,
and its facts are specific enough to verify live.

---

## The walkthrough

### 1. Open — what this is (30 seconds)

> "This is a chat assistant that can answer from my own documents. It runs on
> 18 models across three providers, and it tells you where every answer came
> from."

Point at the sidebar: Chats, Model, Documents, Parameters, Voice, Evaluation.

### 2. It is an ordinary chatbot first (30 seconds)

Type: **`Hello, what can you do?`**

> "Notice there is no sources panel. It did not search — it knew this was
> chitchat. Retrieval only runs when the question calls for it, so ordinary
> conversation costs nothing extra."

### 3. Ask the documents (90 seconds) — the core demo

Type: **`What is the salary range in the job posting?`**

Expected: **$67,616 – $115,913**, and the answer is spoken aloud.

Open the **📄 passages used** panel.

> "Four passages, each with its source file, page number and similarity score.
> This came from page 4 of a scanned PDF — the document has no text layer at
> all, so it was read with OCR before being indexed."

**Then verify it live.** Open `docs/qwerty1.pdf` at page 4 and show the figure.

> "That is the point. Every claim traces to a passage you can check."

### 4. It searches again for a new question (60 seconds)

Type: **`I know Python and Java but only have 1 year of experience. Am I a fit?`**

Expected: the role needs **2+ years**, you have 1, so you fall short — on
languages you qualify.

> "Two things there. It went back and fetched a *different* page for this
> question rather than reusing the salary passages. And it combined what the
> document says with what I told it about myself — it is not limited to
> quoting."

*(If asked: an earlier version reused the old passages and said experience was
not specified. It is specified, on page 3. That is why the prompt now tells it
to search again for a new aspect.)*

### 5. It refuses what it does not know (30 seconds)

Type: **`Does the job mention a signing bonus?`**

Expected: it says the documents do not cover it.

> "This matters more than the right answers. A system that invents a plausible
> answer here is worse than useless, because you cannot tell when to trust it."

### 6. Evaluation — a second model marks the first (90 seconds)

Scroll to the salary answer, click **⚖️ Evaluate this answer**.

Expected: **✅ correct**, judged by Groq.

> "A different model — different provider — is shown the question, the
> passages, and the answer, and asked whether the answer is actually
> supported. One word back: correct or incorrect. Cross-provider on purpose:
> a model marking its own work mostly agrees with itself."

Open the popover to show the judge's raw reply.

### 7. Show the plumbing (60 seconds)

Sidebar → **🔍 Browse chunks** → **Load chunks**.

> "587 chunks. This is how the documents were actually cut up — 700 characters
> each, split on paragraph boundaries."

Filter to `Introduction_to_AI_Agents.pdf` → **1 chunk**.

> "That one is my course certificate, not course material. Ask this system
> what an AI agent is and it correctly says it does not know — the filename
> promises more than the file contains."

Then **📊 What is stored** → **Check index**: `text-embedding-3-small`, 1536
dimensions, cosine, 587 vectors.

### 8. Upload live (60 seconds)

Click the **📎** in the chat box, attach any PDF, ask something only it could
answer.

> "Embedded on upload — no build step. It goes to a temporary folder outside
> the project, and when I clear the session both the file and its vectors are
> deleted. Nothing is kept."

Then **➕ New chat**, ask the same question.

> "The new chat cannot see it. Documents attached to a chat stay in that chat;
> the sidebar uploader is for documents every chat should see."

### 9. Voice (30 seconds)

Click **🎤**, ask a question aloud, press stop, check the transcription, Send.

> "Speech in, speech out. The transcribed question takes exactly the same path
> as a typed one — it still searches, still cites, still gets evaluated."

### 10. Models and parameters (60 seconds)

Switch provider to **Groq** → point at the model list changing.

> "Eighteen models, three providers. Switching mid-conversation keeps the
> history — the memory is not tied to the model."

Open the **ℹ️** beside Parameters.

> "Not every model supports every control. Top K only exists on Gemini. The
> two penalties are rejected by Gemini. Three Groq models cannot call tools at
> all, so they cannot search documents — the app hides what will not work
> rather than letting it fail."

---

## If something goes wrong

| Symptom | Say this, then do this |
|---------|------------------------|
| Rate-limit error | "That is the free tier, not the app." Switch provider |
| No sources panel | The question did not trigger a search. Ask a sharper document question |
| Mic does nothing | Skip it. Chrome permission; not worth debugging live |
| Answer looks wrong | Open the passages and check. **Being able to check is the feature** |

---

## Good closing line

> "The hardest problem here was not getting answers — it was making wrong ones
> visible. Early on it answered a question about a job title using a passage
> from a 10-K filing. It was fluent, well-cited and completely wrong. That is
> why every answer shows its sources, why it refuses stale indexes, and why a
> second model can mark its work."

---

## Questions you may get

**"Which embedding model, and why?"**
`text-embedding-3-small`, 1536 dimensions. Measured against `3-large` and
Gemini on these documents: it separated right from wrong passages by a wider
margin than either, at a sixth of the cost of `3-large`.

**"Why Pinecone?"**
Managed, free tier, and each chunk setting lives in its own namespace, so
switching between chunk sizes is instant once built. Qdrant was considered for
its dashboard; a chunk browser built on our own data turned out to be more
useful and far less disruptive.

**"How much does it cost?"**
Indexing all five documents: about $0.0002. Each question: roughly $0.0005.

**"Does it hallucinate?"**
It can, like anything. The defences are that answers cite passages you can
check, it refuses when the documents do not cover something, and a second
model can mark the answer. None of those is a guarantee — which is exactly why
the sources are always one click away.
