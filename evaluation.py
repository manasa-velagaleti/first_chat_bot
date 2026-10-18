"""Judge a RAG answer with a second model.

The pattern is LLM-as-judge: one model answers from retrieved passages, a
different model is shown the question, those same passages, and the answer,
and says whether the answer is actually supported.

Using a second model matters. Asking the model that wrote an answer to mark
its own work mostly gets agreement - it has already committed to the reasoning
that produced it. A different model, given only the evidence, has no such
stake.

Judging needs plain text generation and no tools, so every model can do it,
including the three that cannot search documents.
"""

from __future__ import annotations

from dataclasses import dataclass

import config as cfg
from chatbot import build_model

# The evaluation prompt, kept verbatim. A judge that can only answer with one
# word is one whose output can be counted; let it explain and you get prose
# that has to be parsed and argued with.
JUDGE_PROMPT = """You are given a question, an answer and reference text. \
You must determine whether the given answer correctly answers the question \
based on the reference text. Here is the data:
[BEGIN DATA]
************
[Question]: {question}
************
[Reference]: {context}
************
[Answer]: {sampled_answer}
[END DATA]
Your response must be a single word, either "correct" or "incorrect", and \
should not contain any text or characters aside from that word. "correct" \
means that the question is correctly and fully answered by the answer. \
"incorrect" means that the question is not correctly or only partially \
answered by the answer"""


@dataclass
class Verdict:
    verdict: str          # "correct" | "incorrect" | "unclear"
    judge: str            # provider label + model
    raw: str              # what the judge actually said
    error: str | None = None


def format_context(sources: list[dict]) -> str:
    """The retrieved passages, as the judge sees them."""
    return "\n\n".join(
        f"[{s.get('n', i + 1)}] ({s.get('file', '?')}"
        + (f" page {s['page']}" if s.get("page") is not None else "")
        + f")\n{s.get('text', '')}"
        for i, s in enumerate(sources)
    )


def judge_answer(question: str, sources: list[dict], answer: str,
                 provider: cfg.Provider, model_name: str) -> Verdict:
    """Ask one model whether another model's answer is supported."""
    label = f"{provider.label} · {model_name}"

    if not sources:
        return Verdict("unclear", label, "",
                       "No retrieved passages to judge against - this answer "
                       "did not come from the documents.")

    prompt = JUDGE_PROMPT.format(
        question=question,
        context=format_context(sources),
        sampled_answer=answer,
    )

    try:
        settings = cfg.default_settings()
        settings["temperature"] = 0.0      # a judge should not be creative
        text = (build_model(provider, model_name, settings)
                .invoke(prompt).text or "")
    except Exception as exc:
        return Verdict("unclear", label, "",
                       cfg.explain_error(exc, provider, model_name))

    # The prompt asks for one word; models sometimes add punctuation or a
    # sentence anyway, so read the answer rather than require the format.
    low = text.strip().lower()
    if low.startswith("correct") or "incorrect" not in low and "correct" in low:
        verdict = "correct"
    elif "incorrect" in low:
        verdict = "incorrect"
    else:
        verdict = "unclear"

    return Verdict(verdict, label, text.strip()[:300])


def default_judge(chat_provider: cfg.Provider) -> tuple[cfg.Provider, str] | None:
    """Pick a judge from a different provider than the one being judged.

    Cross-provider by default because that is the point of the exercise: two
    models from the same family share training and tend to share mistakes.
    """
    others = [p for p in cfg.available_providers() if p.key != chat_provider.key]
    pool = others or cfg.available_providers()
    if not pool:
        return None
    prov = pool[0]
    return prov, prov.models[0]
