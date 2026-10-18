"""RAG evaluation exercise: answer with one model, judge with another.

    python homework_rag_eval.py

Follows the assignment end to end:

    1-2  read the document and extract its text   -> core.load_documents
    3    choose an embedding model                -> text-embedding-3-small
    4    create a vector index                    -> Pinecone
    5-6  query it and get the relevant chunks     -> rag_pinecone
    7    select the answering LLM                 -> gpt-4o-mini
    8-9  build the prompt and get the answer, five times
    then a second model marks each answer correct or incorrect

Steps 1-6 already exist in embeddings_app; this script is the experiment on
top, not another copy of the pipeline.

Writes results to rag_eval_results.txt.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "embeddings_app"))

import config as cfg                    # noqa: E402
import evaluation as ev                 # noqa: E402
import rag_pinecone as rag              # noqa: E402

ANSWER_PROVIDER, ANSWER_MODEL = "openai", "gpt-4o-mini"
NAMESPACE = "chars700"
OUT = ROOT / "rag_eval_results.txt"

# Five questions about the Siemens job posting in docs/. A mix on purpose:
# three plainly answerable, one needing two separate facts, and one whose
# answer is in the document but was mangled by OCR - a judge that marks
# everything "correct" would not catch that last one.
QUESTIONS = [
    "What is the job ID and job title?",
    "What is the salary range for this position?",
    "Which programming languages does the role require?",
    "Which communication protocols should the candidate know?",
    "Does this job offer visa sponsorship?",
]

# Judges from two other providers, so the answering model is never marking
# its own work and we can see whether the judges agree with each other.
JUDGES = [("groq", "openai/gpt-oss-120b"), ("google", "gemini-3.5-flash")]


def main() -> int:
    ans_provider = cfg.PROVIDERS[ANSWER_PROVIDER]
    store = rag.open_store(rag.INDEX_NAME, NAMESPACE)

    lines = [
        "RAG ANSWER EVALUATION",
        f"Run: {datetime.now():%Y-%m-%d %H:%M}",
        f"Answering model: {ans_provider.label} / {ANSWER_MODEL}",
        f"Judges: " + ", ".join(f"{cfg.PROVIDERS[p].label}/{m}" for p, m in JUDGES),
        f"Index: {rag.INDEX_NAME} / {NAMESPACE}",
        "=" * 74,
    ]
    tally: dict[str, dict[str, int]] = {m: {} for _, m in JUDGES}

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"[{i}/{len(QUESTIONS)}] {question}")
        result = rag.ask(store, question, k=4, model=ANSWER_MODEL)
        answer = result["answer"]
        sources = result["sources"]

        lines += ["", f"Q{i}. {question}", "-" * 74,
                  "ANSWER:", answer, "",
                  "RETRIEVED PASSAGES:"]
        for s in sources:
            where = f" p{s['page']}" if s.get("page") is not None else ""
            lines.append(f"  [{s['n']}] {s['file']}{where}  score {s['score']}")
            lines.append("      " + " ".join(s["text"].split())[:200])

        lines.append("")
        lines.append("VERDICTS:")
        for pkey, mname in JUDGES:
            v = ev.judge_answer(question, sources, answer,
                                cfg.PROVIDERS[pkey], mname)
            tally[mname][v.verdict] = tally[mname].get(v.verdict, 0) + 1
            note = f"  ({v.error})" if v.error else ""
            lines.append(f"  {cfg.PROVIDERS[pkey].label}/{mname}: "
                         f"{v.verdict}{note}")
            print(f"      {mname}: {v.verdict}")

    lines += ["", "=" * 74, "SUMMARY"]
    for mname, counts in tally.items():
        summary = ", ".join(f"{k} {n}" for k, n in sorted(counts.items()))
        lines.append(f"  {mname}: {summary}")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWritten to {OUT.name}")
    for mname, counts in tally.items():
        print(f"  {mname}: {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
