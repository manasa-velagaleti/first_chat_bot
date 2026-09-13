"""Single source of truth: what each provider offers and how to pass it.

Providers disagree about sampling parameters in three separate ways, and all
three are handled here so the rest of the app never has to care:

1. Some parameters are simply unsupported  (top_k on OpenAI/Groq)
2. Some are named differently             (max_output_tokens vs max_tokens)
3. Some are supported by the REST API but are not constructor fields on the
   LangChain class, so they must be nested inside `model_kwargs`
   (top_p, penalties and seed on Groq)

Point 3 is the dangerous one: ChatGroq is configured with extra="ignore", so
passing top_p=0.9 straight to the constructor is silently discarded - no error,
no effect. Routing through `native` / `via_model_kwargs` below prevents that.

The sets below were verified against the installed packages by inspecting
<ChatClass>.model_fields. Re-run that check after upgrading a provider package.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

# Loaded here rather than in the entry points, so that has_key() is accurate
# no matter which module imports config first.
load_dotenv()


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamSpec:
    key: str            # our canonical name
    label: str          # shown in the sidebar
    kind: str           # "float" | "int" | "text"
    default: Any
    help: str           # the tooltip: what this does to the output
    summary: str        # one short line, for the comparison table
    min: float | None = None
    max: float | None = None
    step: float | None = None
    # Clickable suggestions. Floats render as a slider + typed box; ints and
    # text render as a picker that also accepts a value you type in.
    presets: list[Any] = field(default_factory=list)


PARAM_SPECS: list[ParamSpec] = [
    ParamSpec(
        key="temperature", label="Temperature", kind="float",
        min=0.0, max=2.0, step=0.05, default=0.5,
        presets=[0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        summary="Randomness of word choice",
        help=(
            "**How random the wording is.**\n\n"
            "At each step the model has a ranked list of likely next words. "
            "Temperature flattens or sharpens that ranking.\n\n"
            "- **0** - always takes the top word. Same question, same answer. "
            "Best for facts, code, extraction, anything you'll test.\n"
            "- **0.3-0.7** - mild variety. Natural conversation.\n"
            "- **1.0-1.5** - noticeably creative, more surprising word choices.\n"
            "- **Above 1.5** - often rambling or incoherent.\n\n"
            "*Try it:* ask \"name a colour\" at 0 five times (identical), "
            "then at 2.0 (different each time)."
        ),
    ),
    ParamSpec(
        key="top_p", label="Top P", kind="float",
        min=0.0, max=1.0, step=0.05, default=1.0,
        presets=[0.1, 0.5, 0.9, 0.95, 1.0],
        summary="Limits the candidate pool by probability",
        help=(
            "**Nucleus sampling - trims the pool of candidate words.**\n\n"
            "Keeps only the most likely words whose probabilities add up to P, "
            "then picks from those.\n\n"
            "- **1.0** - no trimming (off).\n"
            "- **0.9** - drops the unlikely tail. Safer, more predictable.\n"
            "- **0.1** - only the very top candidates. Almost deterministic.\n\n"
            "Similar effect to temperature, achieved differently. **Tune one or "
            "the other, not both** - together they interact confusingly."
        ),
    ),
    ParamSpec(
        key="top_k", label="Top K", kind="int",
        min=1, max=100, step=1, default=None,
        presets=[1, 5, 10, 20, 40, 64],
        summary="Limits the candidate pool by count",
        help=(
            "**Like Top P, but a fixed count instead of a probability share.**\n\n"
            "Only the K most likely words are considered at each step.\n\n"
            "- **1** - always the single best word. Fully deterministic, "
            "ignores temperature entirely.\n"
            "- **10-40** - a normal working range.\n"
            "- **Blank** - the model's own default.\n\n"
            "Gemini only; OpenAI and Groq don't offer it."
        ),
    ),
    ParamSpec(
        key="max_output_tokens", label="Max output tokens", kind="int",
        min=1, max=32768, step=64, default=None,
        presets=[16, 64, 256, 512, 1024, 4096],
        summary="Hard length limit on the reply",
        help=(
            "**A hard ceiling on reply length** (~4 characters per token, so "
            "100 tokens is roughly 75 words).\n\n"
            "This is a guillotine, not a writing instruction - the reply is cut "
            "off **mid-sentence** when it hits the cap. To get genuinely shorter "
            "answers, ask for brevity in your prompt instead.\n\n"
            "- **Blank** - the model's own default.\n"
            "- **16-64** - useful for proving the cap works.\n\n"
            "*Try it:* set 16 and ask for a long explanation - it stops abruptly."
        ),
    ),
    ParamSpec(
        key="frequency_penalty", label="Frequency penalty", kind="float",
        min=-2.0, max=2.0, step=0.1, default=0.0,
        presets=[-2.0, -1.0, 0.0, 0.5, 1.0, 2.0],
        summary="Discourages repeating the same words",
        help=(
            "**Penalises words the more often they've already been used.**\n\n"
            "Scales with the repeat count, so it targets overused words "
            "specifically.\n\n"
            "- **0** - off.\n"
            "- **0.5-1.0** - reduces repetitive phrasing and verbal tics.\n"
            "- **2.0** - strains for synonyms; can read oddly.\n"
            "- **Negative** - encourages repetition, sometimes into a loop.\n\n"
            "*Try it:* \"write 4 lines about the sea\" at -2 (repetitive) vs 2."
        ),
    ),
    ParamSpec(
        key="presence_penalty", label="Presence penalty", kind="float",
        min=-2.0, max=2.0, step=0.1, default=0.0,
        presets=[-2.0, -1.0, 0.0, 0.5, 1.0, 2.0],
        summary="Pushes toward new subject matter",
        help=(
            "**Penalises any word that has appeared at all**, regardless of how "
            "often.\n\n"
            "Frequency penalty fights *repetition*; presence penalty pushes "
            "toward *new topics*.\n\n"
            "- **0** - off.\n"
            "- **0.5-1.0** - broader coverage, more topic variety.\n"
            "- **2.0** - can wander off the question.\n\n"
            "*Try it:* \"list ideas for a party\" at 0 vs 1.5 - the second "
            "roams further."
        ),
    ),
    ParamSpec(
        key="stop", label="Stop sequences", kind="text", default="",
        presets=["END", "###", "\\n", "\\n\\n", "STOP", "Human:", "```"],
        summary="Text that halts generation",
        help=(
            "**Generation stops the instant one of these strings appears.**\n\n"
            "The stop text itself is **not** included in the reply, and the cut "
            "is immediate - usually mid-sentence.\n\n"
            "Pick from the list or type your own; several can be active at once. "
            "Use `\\n` for a newline.\n\n"
            "Handy for structured output - stop at `###` so the model can't "
            "ramble past your delimiter.\n\n"
            "*Try it:* set `THREE`, then ask it to count ONE to FIVE."
        ),
    ),
    ParamSpec(
        key="seed", label="Seed", kind="int",
        min=0, max=2**31 - 1, step=1, default=None,
        presets=[0, 1, 7, 42, 123, 2024],
        summary="Makes sampling reproducible",
        help=(
            "**Makes randomness repeatable.**\n\n"
            "With the same seed, same prompt and same settings, you get the same "
            "reply - which makes it possible to change one parameter and know "
            "the difference came from *that*.\n\n"
            "- **Blank** - fresh randomness every time.\n"
            "- **Any number** - reproducible.\n\n"
            "Best-effort: providers don't guarantee it across model updates. "
            "At temperature 0 output is largely repeatable anyway."
        ),
    ),
]

PARAM_BY_KEY = {spec.key: spec for spec in PARAM_SPECS}

ALL_PARAMS = {spec.key for spec in PARAM_SPECS}


def default_settings() -> dict[str, Any]:
    return {spec.key: spec.default for spec in PARAM_SPECS}


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    env_key: str
    prefix: str                       # init_chat_model provider prefix
    models: list[str]
    native: set[str]                  # passed as direct constructor kwargs
    via_model_kwargs: set[str] = field(default_factory=set)
    renames: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    # Some individual models ignore sampling parameters no matter what the
    # provider as a whole accepts, so capability is per (provider, model).
    fixed_sampling_models: set[str] = field(default_factory=set)
    fixed_sampling_params: set[str] = field(default_factory=set)
    # Models that refuse a request whose *expected* output exceeds a per-minute
    # token limit. With no cap set the provider assumes the model's full default
    # output and rejects the call before generating anything, so we supply one.
    model_caps: dict[str, int] = field(default_factory=dict)
    # Free-tier notes shown in the UI, keyed by model.
    model_notes: dict[str, str] = field(default_factory=dict)

    @property
    def supported(self) -> set[str]:
        return self.native | self.via_model_kwargs

    def supports(self, param_key: str, model_name: str | None = None) -> bool:
        if param_key not in self.supported:
            return False
        if model_name and self.ignores(model_name, param_key):
            return False
        return True

    def ignores(self, model_name: str, param_key: str) -> bool:
        """True if this specific model discards the parameter."""
        base = model_name.lower().rsplit("/", 1)[-1]
        return base in self.fixed_sampling_models and param_key in self.fixed_sampling_params

    def has_key(self) -> bool:
        return bool(os.getenv(self.env_key))


PROVIDERS: dict[str, Provider] = {
    "google": Provider(
        key="google",
        label="Google Gemini",
        env_key="GOOGLE_API_KEY",
        prefix="google_genai",
        # Verified callable with this key. Ordered so the default is a model
        # with workable free-tier quota rather than the most capable one -
        # gemini-3.8-flash is better but allows only 20 requests/day free.
        models=[
            "gemini-3.7-flash",
            "gemini-3.5-flash",
            "gemini-3.8-flash",
            "gemini-3.1-flash-lite",
            "gemini-3-flash-preview",
            "gemini-3.5-flash-lite",
            "gemini-3.1-pro-preview",
        ],
        model_notes={
            "gemini-3.8-flash": "Free tier allows only 20 requests/day for this "
                                "model - expect quota errors while experimenting.",
            "gemini-3.1-pro-preview": "Pro models have a very small free-tier "
                                      "quota and run out quickly.",
        },
        # ChatGoogleGenerativeAI declares every one of our parameters directly.
        native=set(ALL_PARAMS),
        renames={},
        notes="The only provider offering top_k. Uses max_output_tokens natively.",
        # These two run at fixed sampling settings and silently discard
        # temperature/top_k/top_p (enforced inside langchain-google-genai).
        fixed_sampling_models={"gemini-3.5-flash-lite", "gemini-3.6-flash"},
        fixed_sampling_params={"temperature", "top_k", "top_p"},
    ),
    "groq": Provider(
        key="groq",
        label="Groq",
        env_key="GROQ_API_KEY",
        prefix="groq",
        # Confirmed against this account with groq.Groq().models.list().
        # Whisper/prompt-guard/orpheus entries are omitted - they are speech and
        # classifier models, not chat models.
        models=[
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "qwen/qwen3.6-27b",
            "groq/compound",
            "groq/compound-mini",
            "allam-2-7b",
        ],
        # ChatGroq declares only these as real fields.
        native={"temperature", "max_output_tokens", "stop"},
        # Accepted by the Groq REST API but absent from the class, so they must
        # be nested - otherwise extra="ignore" drops them without a word.
        via_model_kwargs={"top_p", "frequency_penalty", "presence_penalty", "seed"},
        renames={"max_output_tokens": "max_tokens"},
        notes="No top_k. Several parameters ride inside model_kwargs.",
        # These two enforce 1000 output-tokens-per-minute. Without a cap Groq
        # assumes the model's full default output (~1044) and returns 429
        # before generating a token, so an unset field would look broken.
        model_caps={
            "qwen/qwen3.8-27b": 800,
            "qwen/qwen3.6-27b": 800,
        },
        model_notes={
            "qwen/qwen3.8-27b": "Limited to 1000 output tokens/minute; capped "
                                "at 800 automatically unless you set your own.",
            "qwen/qwen3.6-27b": "Limited to 1000 output tokens/minute; capped "
                                "at 800 automatically unless you set your own.",
        },
    ),
    "openai": Provider(
        key="openai",
        label="OpenAI",
        env_key="OPENAI_API_KEY",
        prefix="openai",
        models=[
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4o-mini",
        ],
        native={
            "temperature", "top_p", "max_output_tokens",
            "frequency_penalty", "presence_penalty", "stop", "seed",
        },
        renames={"max_output_tokens": "max_tokens"},
        notes="No top_k. Reasoning (o-series) models ignore temperature.",
    ),
}


def available_providers() -> list[Provider]:
    """Only the providers whose API key is actually set."""
    return [p for p in PROVIDERS.values() if p.has_key()]


def parse_stop(raw: str | list[str] | None) -> list[str] | None:
    """Normalise stop sequences to a list the provider will accept.

    Accepts either a list (from the multiselect) or a comma-separated string,
    and turns a typed '\\n' into a real newline - people write the escape, but
    the API needs the character.
    """
    if raw is None:
        return None

    if isinstance(raw, str):
        items = [s.strip() for s in raw.split(",")]
    else:
        items = [str(s).strip() for s in raw]

    out = []
    for item in items:
        if not item:
            continue
        out.append(item.replace("\\n", "\n").replace("\\t", "\t"))
    return out or None


def translate(
    provider: Provider,
    settings: dict[str, Any],
    model_name: str | None = None,
) -> dict[str, Any]:
    """Turn UI settings into constructor kwargs for this provider.

    Unsupported parameters are dropped, names are mapped, and anything the
    LangChain class does not declare is nested under model_kwargs. Blank values
    are omitted entirely so the provider's own default applies. Passing
    model_name also drops params that this particular model would discard.
    """
    direct: dict[str, Any] = {}
    extra: dict[str, Any] = {}

    for spec in PARAM_SPECS:
        value = settings.get(spec.key)

        if spec.key == "stop":
            value = parse_stop(value)

        if value is None or value == "" or value == []:
            continue
        if not provider.supports(spec.key, model_name):
            continue

        name = provider.renames.get(spec.key, spec.key)

        if spec.key in provider.native:
            direct[name] = value
        elif spec.key in provider.via_model_kwargs:
            extra[name] = value

    # Supply a default output cap for models that reject uncapped requests.
    cap = provider.model_caps.get(model_name or "")
    if cap and not settings.get("max_output_tokens"):
        direct[provider.renames.get("max_output_tokens", "max_output_tokens")] = cap

    if extra:
        direct["model_kwargs"] = extra

    return direct


def model_note(provider: Provider, model_name: str) -> str | None:
    """Free-tier or per-model caveat worth showing beside the picker."""
    return provider.model_notes.get(model_name)


def explain_error(err: Exception, provider: Provider, model_name: str) -> str:
    """Turn a raw provider exception into something worth acting on.

    Raw SDK errors are long and bury the one sentence that matters, so the
    common failures get a plain reading plus the actual next step.
    """
    text = str(err)
    low = text.lower()

    if "not_found" in low or "does not exist" in low or "404" in text:
        return (
            f"**`{model_name}` isn't available on your {provider.label} key.**\n\n"
            "Model IDs change often. Pick another from the dropdown, or type a "
            "current one - the field accepts any ID."
        )

    if "otpm" in low or "request too large" in low:
        return (
            f"**`{model_name}` has a per-minute output-token limit** that this "
            "request would exceed before generating anything.\n\n"
            "Set **Max output tokens** to 800 or less and try again."
        )

    if "429" in text or "resource_exhausted" in low or "quota" in low:
        wait = ""
        match = re.search(r"retry in ([\d.]+)s", text, re.IGNORECASE)
        if match:
            wait = f" Retry in about {float(match.group(1)):.0f} seconds."
        return (
            f"**Rate limit / quota reached on `{model_name}`.**{wait}\n\n"
            "Free tiers are small and per-model. Either wait, or switch to "
            "another model in the dropdown - the quota is counted separately "
            "for each one."
        )

    if "api key" in low or "401" in text or "unauthorized" in low or "permission" in low:
        return (
            f"**{provider.label} rejected the API key.**\n\n"
            f"Check `{provider.env_key}` in your .env file, then restart the app."
        )

    return f"**Request failed:** {text[:400]}"


def support_rows() -> list[dict[str, Any]]:
    """Data for the help popover: one row per parameter, one column per provider."""
    rows = []
    for spec in PARAM_SPECS:
        row = {"Parameter": spec.label}
        for p in PROVIDERS.values():
            row[p.label] = "Yes" if p.supports(spec.key) else "No"
        row["What it does"] = spec.summary
        rows.append(row)
    return rows


def fixed_sampling_note(provider: Provider, model_name: str) -> str | None:
    """Warn text when the chosen model ignores some parameters it otherwise has."""
    ignored = sorted(
        PARAM_BY_KEY[k].label
        for k in provider.supported
        if provider.ignores(model_name, k)
    )
    if not ignored:
        return None
    return (
        f"`{model_name}` runs at fixed sampling settings, so "
        + ", ".join(ignored)
        + " are hidden - this model would ignore them."
    )
