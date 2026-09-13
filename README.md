# Multi-Model Chat Assistant

A chat app for **comparing language models side by side**. Switch provider and model
mid-conversation, tune all eight sampling parameters live, and watch how each one
changes the reply — with the conversation kept intact the whole time.

Built on LangChain 1.0 + LangGraph, with a Streamlit interface.

```
┌──────────────────────────┬──────────────────────────────────────────────┐
│ Provider  [Google ▾]     │  💬 Chat Assistant                           │
│ Model     [3.7-flash ▾]  │  Google Gemini · gemini-3.7-flash            │
│                          │                                              │
│ Parameters          (ℹ️) │   You:  My name is Srija                     │
│                          │   Bot:  Nice to meet you, Srija!             │
│ Temperature   [0.50]     │         Google Gemini · gemini-3.7-flash     │
│ ──────●───────           │                                              │
│ (0)(0.3)(0.5)(0.7)(1)(2) │   You:  What's my name?      ← after moving  │
│                          │   Bot:  Your name is Srija.     the slider   │
│ Top P         [1.00]     │                                              │
│ ───────────●             │                                              │
│                          │                                              │
│ Top K      [default ▾]   │                                              │
│ Max tokens [default ▾]   │                                              │
│ Stop seqs  [ ×END      ] │                                              │
│ Seed       [42        ▾] │                                              │
│                          │                                              │
│ [Reset params][New chat] │  ┌────────────────────────────────────────┐  │
│ ▸ Exactly what gets sent │  │ Ask me anything...                     │  │
└──────────────────────────┴──└────────────────────────────────────────┘──┘
```

---

## What it does

**1. Model switching.** Three providers, 18 models. The provider list builds itself
from which API keys are in `.env` — no key, no entry, so you can never pick something
that will fail on auth.

**2. Live parameter control.** All eight sampling parameters, each with a slider *and*
a typed box *and* clickable presets. Change one, send a message, see the difference.

**3. Honest capability handling.** Not every model accepts every parameter. Controls
that wouldn't apply are hidden rather than silently ignored, and an ℹ️ panel explains
which model supports what.

**4. Memory that survives everything.** Change a parameter, switch models, switch
providers — the conversation carries on. This is the hard part, and it's the reason
the code is shaped the way it is.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add at least one key to `.env`:

| Provider | Key | Where | Cost |
|----------|-----|-------|------|
| Google Gemini | `GOOGLE_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Free tier |
| Groq | `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) | Free tier |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) | Paid |

```bash
.venv\Scripts\python.exe -m streamlit run app.py    # the app
.venv\Scripts\python.exe cli.py                     # terminal version
.venv\Scripts\python.exe test_params.py             # verify everything works
```

> **Editing `config.py` or `chatbot.py` requires restarting the server.** Streamlit
> hot-reloads `app.py` but keeps imported modules cached, so changes to the others
> won't appear until you restart.

---

## Screens

### The main view

![Main view](docs/img/01-overview.png)

Sidebar on the left holds everything that shapes the request; chat on the right. Each
reply is captioned with the model and temperature that produced it, so a transcript
where you switched models stays readable afterwards.

### Parameters adapt to the model

![Parameter controls](docs/img/02-parameters.png)

Every parameter offers three ways to set it — drag the slider, type an exact number,
or click a preset. Select OpenAI or Groq and **Top K disappears entirely**, because
neither accepts it.

### The ℹ️ support table

![Support table](docs/img/03-help.png)

Rendered from the same data structure that drives the controls, so it can't fall out
of sync with what the app actually does. Below the table sits a full description of
every parameter.

### Exactly what gets sent

![Payload preview](docs/img/04-payload.png)

Expand this to see the literal keyword arguments handed to the provider. It's the
fastest way to confirm a parameter is really being applied — and it makes the
provider differences concrete:

```jsonc
// Gemini - everything native
{"temperature": 0.5, "top_k": 40, "max_output_tokens": 256, "seed": 42}

// Groq - renamed, and half of it nested
{"temperature": 0.5, "max_tokens": 256,
 "model_kwargs": {"top_p": 1.0, "seed": 42}}
```

> **Screenshots not generated yet.** They need a headless browser, and this machine's
> C: drive is currently full. Either free some space and run
> `python docs/capture_screenshots.py`, or take them yourself with **Win+Shift+S** and
> save as `docs/img/01-overview.png` … `04-payload.png`.

---

## How it works

### The pieces

| File | Role |
|------|------|
| `config.py` | **The brain.** Providers, models, parameter specs, capability matrix |
| `chatbot.py` | Builds agents, holds the shared memory, streams replies |
| `app.py` | The Streamlit UI — renders itself from `config.py` |
| `cli.py` | Terminal version, defaults only |
| `test_params.py` | Automated proof that parameters actually reach the models |
| `TESTING.md` | Manual test cases + known quirks |

### The one design decision that matters

`create_agent()` bakes the model into the agent. So **every parameter change requires
building a new agent** — and a naive implementation loses the conversation each time
you nudge a slider.

The fix is to give the agent and the memory different lifetimes:

```python
# chatbot.py - one module-level store, never rebuilt
_CHECKPOINTER = MemorySaver()

def build_agent(provider, model_name, settings):
    return create_agent(
        model=build_model(provider, model_name, settings),  # new every time
        checkpointer=get_checkpointer(),                    # always the same
        ...
    )
```

History lives in the checkpointer keyed by `thread_id`, not in the agent. Rebuild the
agent as often as you like; the conversation doesn't notice. This is also why
switching *models* mid-chat works — messages are stored provider-neutral, so the next
model just receives the earlier turns as context.

### Three ways providers disagree

This is the messy reality the app exists to hide, all handled in `config.py`:

**1. Unsupported.** `top_k` doesn't exist on OpenAI or Groq. Those controls are hidden.

**2. Renamed.** Gemini wants `max_output_tokens`; OpenAI and Groq want `max_tokens`.

**3. Nested.** Groq accepts `top_p`, both penalties and `seed` at the REST level, but
`ChatGroq` doesn't declare them as fields — and it's configured with `extra="ignore"`.
Passing them directly is **silently discarded**: no error, no effect, a slider that
does nothing. They're routed through `model_kwargs` instead.

There's a fourth case that's per-*model* rather than per-provider: `gemini-3.5-flash-lite`
and `gemini-3.6-flash` run at fixed sampling and ignore Temperature/Top K/Top P even
though Gemini supports them. Selecting one hides those three and says why.

### Everything renders from one list

The sidebar, the tooltips and the support table are all generated by looping over
`PARAM_SPECS`. Adding a ninth parameter means adding one entry — no UI code changes.

---

## The eight parameters

| Parameter | Range | Default | What it changes |
|-----------|-------|---------|-----------------|
| Temperature | 0–2 | 0.5 | Randomness. 0 ≈ same answer every time |
| Top P | 0–1 | 1.0 | Trims candidates by probability mass |
| Top K | ≥1 | unset | Trims candidates by count. **Gemini only** |
| Max output tokens | ≥1 | unset | Hard length cap — cuts off mid-sentence |
| Frequency penalty | −2–2 | 0 | Discourages repeating words |
| Presence penalty | −2–2 | 0 | Pushes toward new topics |
| Stop sequences | list | empty | Halts generation at these strings |
| Seed | int | unset | Makes sampling reproducible |

Hover any control's ⓘ for a fuller explanation with a suggested experiment.

---

## Verified status

From `test_params.py` plus a sweep of every model:

**Models: 16 of 18 working.** Groq 7/7, OpenAI 4/4, Google 5/7 — the two Gemini
failures are free-tier quota, not defects.

**Parameters: zero failures on all three providers.** The suite distinguishes *"our
code dropped it"* (FAIL) from *"it was sent and the model ignored it"* (WARN) by
inspecting the actual payload. No FAILs anywhere.

### Quirks found while testing

| Thing | Explanation |
|-------|-------------|
| `gemini-3.8-flash` returns 429 | **20 requests/day** free tier, counted per model. Switch models to keep working |
| Gemini 3.x isn't deterministic at temperature 0 | Measured: temp 0, top_p 0.05, top_k 1 and a fixed seed all still varied. Reasoning models. **Use Groq or OpenAI when you need repeatability** |
| Groq ignores `presence_penalty` on `gpt-oss-120b` | Model-side — confirmed against Groq's own SDK |
| Very low max-tokens gives an *empty* reply | Reasoning models spend the budget thinking before emitting text |
| qwen models needed an output cap | 1000 output-tokens/minute limit; the app now applies 800 automatically |

---

## Extending it

**A ninth parameter** — add a `ParamSpec` to `PARAM_SPECS` and list its key in the
providers that accept it.

**A new provider** — add a `Provider` entry. Verify its capabilities rather than
guessing:

```python
from langchain_groq import ChatGroq
print(sorted(ChatGroq.model_fields))   # what it really accepts
```

**Tools / function calling** — define them with `@tool` and pass to
`create_agent(tools=[...])` in `chatbot.py`.

**Persistent history** — swap `MemorySaver` for
`langgraph.checkpoint.sqlite.SqliteSaver`.

**Tracing** — uncomment the LangSmith keys in `.env` to see every call, token count
and latency at smith.langchain.com.

> **Model IDs go stale.** Groq in particular retires models often. Every model
> dropdown accepts a typed ID, so a stale list never blocks you.
