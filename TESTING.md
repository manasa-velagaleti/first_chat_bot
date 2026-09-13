# Testing Guide

Two ways to check things work: an automated sweep, and manual tests in the UI.

---

## 1. Automated sweep

```bash
.venv\Scripts\python.exe test_params.py --list          # plan only, no API calls
.venv\Scripts\python.exe test_params.py                 # first model of each provider
.venv\Scripts\python.exe test_params.py --provider groq
.venv\Scripts\python.exe test_params.py --all-models    # every model in every list
```

Each check sends a small request and inspects the reply for evidence the parameter
took effect. Paced at 2s between calls with exponential backoff, because free tiers
rate-limit hard.

### Reading the results

| Status | Meaning | Action |
|--------|---------|--------|
| `PASS` | The parameter demonstrably changed the output | None |
| `FAIL` | The parameter is **missing from the outgoing payload** | Real bug — our code is dropping it |
| `WARN` | Sent correctly, but no observable effect | Usually the model, not the code |
| `SKIP` | Provider/model doesn't support it | Expected |
| `ERR` | The request itself failed | Check key, model ID, or rate limit |

**The `FAIL` / `WARN` split is the important part.** When a check sees no difference
in output, it inspects the actual payload before judging:

- **in the payload** → `WARN "sent in the payload, but this model ignores it"`
- **not in the payload** → `FAIL "our code is dropping it"`

Only `FAIL` is our bug. This distinction exists because Groq's `gpt-oss-120b` genuinely
ignores `presence_penalty` — verified by calling Groq's own SDK directly, where 0.0,
2.0 and −2.0 all return near-identical text.

`WARN` on the sampling checks is probabilistic. One is weak evidence; every run is a
finding.

---

## 2. Manual tests in the UI

Start the app, then work down this list. Expand **"Exactly what gets sent"** at the
bottom of the sidebar to see the real payload as you change things.

### A. Provider and model switching

| # | Do this | Expect |
|---|---------|--------|
| A1 | Open the Provider dropdown | All three providers with keys in `.env` |
| A2 | Switch to Groq | Model list changes to Groq's models |
| A3 | Type a nonsense model ID into Model | Send a message → clear red error, app survives |
| A4 | Select `gemini-3.8-flash`, send "hello" | Reply streams in word by word |

### B. Parameter visibility (the capability matrix)

| # | Do this | Expect |
|---|---------|--------|
| B1 | Gemini selected | **Top K is visible** |
| B2 | Switch to OpenAI or Groq | **Top K disappears** |
| B3 | Select `gemini-3.5-flash-lite` | Temperature, Top P, Top K all vanish + ⚠️ notice |
| B4 | Click the ℹ️ beside "Parameters" | Support table + full description of each parameter |
| B5 | Hover any slider's ⓘ | Tooltip explaining the effect on output |

### C. The controls themselves

| # | Do this | Expect |
|---|---------|--------|
| C1 | Type `1.75` in the box beside Temperature | Slider jumps to 1.75 |
| C2 | Drag the slider | Box updates to match |
| C3 | Click a preset pill under Temperature | Both slider and box jump to it |
| C4 | Open Top K | Preset list: 1, 5, 10, 20, 40, 64 |
| C5 | Type `77` into Top K | Accepted; still there after the next message |
| C6 | Type `abc` into Max output tokens | Warning, value ignored, no crash |
| C7 | Open Stop sequences | Presets: END, ###, \n, \n\n, STOP, Human:, ``` |
| C8 | Pick two stop sequences | Both appear in "Exactly what gets sent" |
| C9 | Set Top K to "Model default" | Disappears from the payload entirely |
| C10 | Click "Reset params" | Everything back to defaults |

### D. Parameters actually affecting output

| # | Setup | Prompt | Expect |
|---|-------|--------|--------|
| D1 | Temperature **0** | "Name one animal, one word" | Same answer every time |
| D2 | Temperature **2.0** | Same | Different answer most times |
| D3 | Max output tokens **64** | "Explain how an engine works in detail" | Cut off mid-sentence |
| D4 | Stop = `THREE` | "Output exactly: ONE TWO THREE FOUR FIVE" | Stops before THREE |
| D5 | Seed **42**, temp **1.0** | "Invent a spaceship name" | Same name twice in a row |
| D6 | Frequency penalty **−2** vs **+2** | "Write 4 lines about the sea" | −2 repetitive, +2 varied |
| D7 | Top K **1**, temperature **2** | "Invent a spaceship name" | Identical despite high temp |

> **D3 caveat:** on reasoning models (Gemini 3.x, Groq's gpt-oss) a *very* low cap can
> return an **empty** reply — the budget is spent thinking before any visible text.
> That's the cap working, not a bug. Use 64+ to see a truncated sentence.

### E. Conversation memory — the important one

| # | Do this | Expect |
|---|---------|--------|
| E1 | "My name is Manasa" | Acknowledges |
| E2 | **Move the temperature slider**, then ask "What's my name?" | **Still knows.** A new model is built on every settings change; the memory is deliberately kept separate |
| E3 | Switch provider entirely, ask again | Still knows — history is provider-neutral |
| E4 | Click "New chat", ask again | Does **not** know — fresh thread |

### F. Error handling

| # | Do this | Expect |
|---|---------|--------|
| F1 | Blank every key in `.env`, restart | Clear message naming the keys to set |
| F2 | Set one key only | Only that provider in the dropdown |
| F3 | Fire many messages fast on a free tier | Rate-limit error shown in red; app still usable |

---

## Known quirks (not bugs)

| Thing | Why |
|-------|-----|
| Groq ignores `presence_penalty` on `gpt-oss-120b` | Model-side. Verified against Groq's own SDK |
| Very low max tokens → empty reply | Reasoning models spend the budget before emitting text |
| `gemini-3.5-flash-lite` ignores temperature/top_p/top_k | Fixed sampling; enforced in `langchain-google-genai` |
| Groq model IDs differ from most tutorials | Groq retires models often — the list here came from your own key |
| Seed not perfectly reproducible | Providers describe it as best-effort |
| Gemini free tier 429s quickly | **20 requests/day per model** on `gemini-3.8-flash`. Counted per model, so switching models in the dropdown gets you working again immediately |
| Gemini 3.x isn't deterministic even at temperature 0 | Measured: temp 0, top_p 0.05, top_k 1 and a fixed seed all still varied on `gemini-3.7-flash`. These are reasoning models and their sampling isn't strictly reproducible. **Groq and OpenAI passed all four of these checks** — use one of those when you need repeatability |
| Groq qwen models need an output cap | 1000 output-tokens/minute limit. Without a cap Groq assumes the model's full default output and rejects the request before generating. The app now applies 800 automatically |
