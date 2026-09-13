"""Automated checks that every parameter actually reaches every model.

Run:
    .venv\\Scripts\\python.exe test_params.py                # all providers
    .venv\\Scripts\\python.exe test_params.py --provider groq
    .venv\\Scripts\\python.exe test_params.py --model gemini-3.8-flash
    .venv\\Scripts\\python.exe test_params.py --list         # no API calls

Each check sends a small, cheap request and inspects the reply for evidence the
parameter took effect. Roughly 13 calls per model.

Result meanings:
    PASS  the parameter demonstrably changed the output
    FAIL  it did not - the value is being dropped somewhere
    SKIP  this provider/model doesn't support the parameter
    WARN  a soft check that didn't hold; model-dependent, not necessarily broken
    ERR   the request itself failed (network, bad model ID, rate limit)

WARN matters most for the sampling checks: "temperature 2 gives varied output"
is probabilistic. A single WARN is weak evidence; a WARN every run is a finding.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import config as cfg
from chatbot import build_agent, reply

warnings.filterwarnings("ignore")

PASS, FAIL, SKIP, WARN, ERR = "PASS", "FAIL", "SKIP", "WARN", "ERR "

_thread = 0
DELAY = 2.0          # seconds between calls; free tiers rate-limit aggressively
MAX_RETRIES = 4


def _is_rate_limit(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(s in text for s in ("429", "rate", "resource_exhausted", "quota"))


def ask(provider, model, settings, prompt) -> str:
    """One request on a fresh thread, so no check sees another's history.

    Retries on rate limits with exponential backoff - the free tiers throttle
    hard enough that an un-paced sweep reports failures that are really 429s.
    """
    global _thread
    _thread += 1

    delay = DELAY
    for attempt in range(MAX_RETRIES):
        try:
            agent = build_agent(provider, model, settings)
            out = reply(agent, prompt, f"probe-{_thread}").strip()
            time.sleep(DELAY)
            return out
        except Exception as exc:  # noqa: BLE001
            if not _is_rate_limit(exc) or attempt == MAX_RETRIES - 1:
                raise
            delay *= 2
            print(f"       (rate limited, waiting {delay:.0f}s)", flush=True)
            time.sleep(delay)
    return ""


def was_sent(provider, model, key, value) -> bool:
    """Is this parameter actually in the payload we hand the provider?

    This is the difference between 'our code dropped it' and 'the model
    received it and ignored it' - two very different bugs, and only the first
    one is ours. Checks both the top-level kwargs and the model_kwargs nest.
    """
    payload = cfg.translate(provider, base(**{key: value}), model)
    native = provider.renames.get(key, key)
    return native in payload or native in payload.get("model_kwargs", {})


def base(**overrides):
    s = cfg.default_settings()
    s.update(overrides)
    return s


# --- individual checks -----------------------------------------------------
# Each returns (status, detail).

def check_connectivity(p, m):
    # No token cap here: Gemini 3.x spends budget on internal reasoning first,
    # so a tight cap can return an empty string and look like a dead provider.
    out = ask(p, m, base(temperature=0.0),
              "Reply with exactly one word: banana")
    if "banana" in out.lower():
        return PASS, f"replied {out[:30]!r}"
    return WARN, f"reachable but said {out[:40]!r}"


def check_max_tokens(p, m):
    long_prompt = "Explain how a car engine works in full detail."
    cap = 400
    capped = ask(p, m, base(temperature=0.0, max_output_tokens=cap), long_prompt)
    uncapped = ask(p, m, base(temperature=0.0, max_output_tokens=4000), long_prompt)

    if not capped:
        # Reasoning models spend the budget thinking before emitting anything,
        # so a tight cap yields an empty string rather than a truncated one.
        return WARN, "cap gave an empty reply - budget consumed by reasoning"

    # Judge against the cap, not against the uncapped run: if the model would
    # naturally have stopped early anyway, comparing the two proves nothing.
    est_tokens = len(capped) / 4
    if est_tokens > cap * 1.3:
        return FAIL, f"~{est_tokens:.0f} tokens returned for a {cap}-token cap"
    if len(capped) >= len(uncapped):
        return WARN, (f"within the cap (~{est_tokens:.0f} tokens) but the "
                      f"uncapped run was no longer - inconclusive")
    return PASS, (f"~{est_tokens:.0f} tokens under a {cap} cap; "
                  f"{len(capped)} vs {len(uncapped)} chars uncapped")


def check_stop(p, m):
    prompt = "Output exactly this and nothing else: ONE TWO THREE FOUR FIVE"
    s = base(temperature=0.0, max_output_tokens=800)
    without = ask(p, m, s, prompt)
    with_stop = ask(p, m, base(**{**s, "stop": ["THREE"]}), prompt)

    if "THREE" in with_stop.upper():
        return FAIL, f"stop ignored, got {with_stop[:40]!r}"
    if with_stop and with_stop != without:
        return PASS, f"halted at {with_stop[:28]!r}"
    if not with_stop and without:
        # On reasoning models the stop string can appear in the hidden thinking
        # and cut the turn before any visible text. Still proof it took effect.
        return PASS, f"truncated to empty (baseline was {len(without)} chars)"
    return WARN, f"inconclusive: with={with_stop[:20]!r} without={without[:20]!r}"


def check_temperature_zero(p, m):
    prompt = "Name one animal. Reply with the single word only."
    runs = [ask(p, m, base(temperature=0.0, max_output_tokens=200), prompt)
            for _ in range(3)]
    if len(set(runs)) == 1:
        return PASS, f"3/3 identical ({runs[0][:20]!r})"
    return WARN, f"varied at temp 0: {sorted(set(runs))[:3]}"


def check_temperature_high(p, m):
    prompt = "Invent a surprising name for a spaceship. Name only."
    runs = [ask(p, m, base(temperature=2.0, top_p=1.0, max_output_tokens=200), prompt)
            for _ in range(3)]
    distinct = len(set(runs))
    if distinct >= 2:
        return PASS, f"{distinct}/3 distinct"
    return WARN, f"all 3 identical at temp 2 ({runs[0][:24]!r})"


def check_seed(p, m):
    prompt = "Invent a surprising name for a spaceship. Name only."
    s = base(temperature=1.0, seed=424242, max_output_tokens=200)
    a, b = ask(p, m, s, prompt), ask(p, m, s, prompt)
    if a == b:
        return PASS, f"reproducible ({a[:24]!r})"
    return WARN, f"differed: {a[:20]!r} vs {b[:20]!r} (providers call seed best-effort)"


def check_top_k(p, m):
    prompt = "Invent a surprising name for a spaceship. Name only."
    s = base(temperature=2.0, top_k=1, max_output_tokens=200)
    runs = [ask(p, m, s, prompt) for _ in range(2)]
    if len(set(runs)) == 1:
        return PASS, f"top_k=1 forced determinism ({runs[0][:22]!r})"
    return WARN, f"top_k=1 still varied: {sorted(set(runs))}"


def _repetition(text: str) -> float:
    """Share of words that are repeats. Higher means more repetitive."""
    words = [w.lower().strip(".,!?;:") for w in text.split() if w.strip()]
    if len(words) < 8:
        return 0.0
    return 1 - len(set(words)) / len(words)


def check_frequency_penalty(p, m):
    prompt = "Write four short lines about the sea. Use the word 'sea' often."
    low = ask(p, m, base(temperature=0.4, frequency_penalty=-1.5,
                         max_output_tokens=300, seed=5), prompt)
    high = ask(p, m, base(temperature=0.4, frequency_penalty=1.8,
                          max_output_tokens=300, seed=5), prompt)
    r_low, r_high = _repetition(low), _repetition(high)
    if low != high:
        verdict = PASS if r_high <= r_low else WARN
        return verdict, f"repetition {r_low:.2f} at -1.5 vs {r_high:.2f} at +1.8"
    if was_sent(p, m, "frequency_penalty", 1.8):
        return WARN, "sent in the payload, but this model ignores it"
    return FAIL, "not present in the outgoing payload - our code is dropping it"


def check_presence_penalty(p, m):
    prompt = "List ideas for a birthday party."
    a = ask(p, m, base(temperature=0.4, presence_penalty=0.0,
                       max_output_tokens=300, seed=5), prompt)
    b = ask(p, m, base(temperature=0.4, presence_penalty=1.8,
                       max_output_tokens=300, seed=5), prompt)
    if a != b:
        return PASS, "output changed between 0.0 and 1.8"
    if was_sent(p, m, "presence_penalty", 1.8):
        return WARN, "sent in the payload, but this model ignores it"
    return FAIL, "not present in the outgoing payload - our code is dropping it"


def check_top_p(p, m):
    prompt = "Invent a surprising name for a spaceship. Name only."
    tight = [ask(p, m, base(temperature=1.5, top_p=0.05, max_output_tokens=200), prompt)
             for _ in range(2)]
    if len(set(tight)) == 1:
        return PASS, f"top_p=0.05 narrowed to one answer ({tight[0][:22]!r})"
    return WARN, f"top_p=0.05 still varied: {sorted(set(tight))}"


CHECKS = [
    ("connectivity",      None,                 check_connectivity),
    ("max_output_tokens", "max_output_tokens",  check_max_tokens),
    ("stop",              "stop",               check_stop),
    ("temperature (0)",   "temperature",        check_temperature_zero),
    ("temperature (2)",   "temperature",        check_temperature_high),
    ("top_p",             "top_p",              check_top_p),
    ("top_k",             "top_k",              check_top_k),
    ("seed",              "seed",               check_seed),
    ("frequency_penalty", "frequency_penalty",  check_frequency_penalty),
    ("presence_penalty",  "presence_penalty",   check_presence_penalty),
]


def run_model(provider, model, only=None) -> dict[str, int]:
    print(f"\n{'=' * 74}\n{provider.label}  ·  {model}\n{'=' * 74}")
    tally = {PASS: 0, FAIL: 0, SKIP: 0, WARN: 0, ERR: 0}

    for name, param_key, fn in CHECKS:
        if only and name not in only:
            continue
        if param_key and not provider.supports(param_key, model):
            print(f"  {SKIP}  {name:<20} not supported by this provider/model")
            tally[SKIP] += 1
            continue
        try:
            status, detail = fn(provider, model)
        except Exception as exc:  # noqa: BLE001 - report, never abort the sweep
            status, detail = ERR, f"{type(exc).__name__}: {str(exc)[:90]}"
        print(f"  {status}  {name:<20} {detail}")
        tally[status] += 1

    return tally


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", help="only this provider key (google/groq/openai)")
    ap.add_argument("--model", help="only this model ID")
    ap.add_argument("--check", action="append", help="only this check (repeatable)")
    ap.add_argument("--all-models", action="store_true",
                    help="every model in the list, not just the first")
    ap.add_argument("--list", action="store_true", help="show the plan, call nothing")
    args = ap.parse_args()

    provs = cfg.available_providers()
    if args.provider:
        provs = [p for p in provs if p.key == args.provider]
    if not provs:
        print("No providers with an API key set. Check your .env file.")
        return 1

    jobs = []
    for p in provs:
        models = p.models if args.all_models else p.models[:1]
        if args.model:
            models = [args.model] if args.model in p.models or args.provider else []
        jobs += [(p, m) for m in models]

    if args.list:
        print("Would test:")
        for p, m in jobs:
            print(f"  {p.label:<15} {m}")
        n = len([c for c in CHECKS if not args.check or c[0] in args.check])
        print(f"\n{len(jobs)} model(s) x ~{n} checks - roughly {len(jobs) * n * 2} API calls.")
        return 0

    totals = {PASS: 0, FAIL: 0, SKIP: 0, WARN: 0, ERR: 0}
    for p, m in jobs:
        for k, v in run_model(p, m, args.check).items():
            totals[k] += v

    print(f"\n{'=' * 74}")
    print(f"TOTAL   pass {totals[PASS]}   fail {totals[FAIL]}   "
          f"warn {totals[WARN]}   skip {totals[SKIP]}   err {totals[ERR]}")
    print("=" * 74)
    if totals[FAIL] or totals[ERR]:
        print("FAIL means a parameter was ignored; ERR means the request failed.")
        return 1
    if totals[WARN]:
        print("WARN checks are probabilistic - re-run before treating one as a bug.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
