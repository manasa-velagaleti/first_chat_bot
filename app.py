"""Multi-model chat assistant.  Run:  streamlit run app.py"""

from __future__ import annotations

from typing import Any

import streamlit as st

import config as cfg
from chatbot import build_agent, reset_thread, stream_reply

st.set_page_config(page_title="Chat Assistant", page_icon="💬", layout="wide")

providers = cfg.available_providers()
if not providers:
    st.error(
        "No API keys found. Add at least one of "
        + ", ".join(p.env_key for p in cfg.PROVIDERS.values())
        + " to your .env file and restart."
    )
    st.stop()

DEFAULT_LABEL = "Model default"

# --- session state ---------------------------------------------------------
if "settings" not in st.session_state:
    st.session_state.settings = cfg.default_settings()
if "history" not in st.session_state:
    st.session_state.history = []          # [(role, text, caption)]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "chat-1"


# --- widget builders -------------------------------------------------------
def render_float(spec: cfg.ParamSpec, base: str, current: Any) -> float:
    """Slider plus a typed box, kept in sync.

    Two widgets can't share one session_state key, so each has its own and an
    on_change callback writes across. Dragging updates the box and typing
    updates the slider.
    """
    sld, num = f"{base}_sld", f"{base}_num"
    start = float(current if current is not None else spec.default)
    st.session_state.setdefault(sld, start)
    st.session_state.setdefault(num, start)

    def slider_changed():
        st.session_state[num] = st.session_state[sld]

    def box_changed():
        st.session_state[sld] = st.session_state[num]

    left, right = st.columns([2, 1], vertical_alignment="bottom")
    with left:
        st.slider(
            spec.label,
            min_value=float(spec.min), max_value=float(spec.max),
            step=float(spec.step), key=sld,
            on_change=slider_changed, help=spec.help,
        )
    with right:
        st.number_input(
            f"{spec.label} value",
            min_value=float(spec.min), max_value=float(spec.max),
            step=float(spec.step), key=num,
            on_change=box_changed, label_visibility="hidden",
        )

    if spec.presets:
        picked = st.pills(
            f"{spec.label} presets",
            spec.presets,
            format_func=lambda v: f"{v:g}",
            key=f"{base}_pills",
            label_visibility="collapsed",
        )
        if picked is not None and picked != st.session_state[sld]:
            st.session_state[sld] = float(picked)
            st.session_state[num] = float(picked)
            st.rerun()

    return float(st.session_state[sld])


def render_int(spec: cfg.ParamSpec, base: str, current: Any) -> int | None:
    """Preset picker that also accepts a number you type in.

    Everything is handled as strings so the 'Model default' sentinel can live
    in the same list as the numbers without mixed-type comparisons.
    """
    options = [DEFAULT_LABEL] + [str(v) for v in spec.presets]
    label = DEFAULT_LABEL if current is None else str(current)
    if label not in options:
        options.append(label)

    choice = st.selectbox(
        spec.label,
        options,
        index=options.index(label),
        accept_new_options=True,
        help=spec.help,
        key=f"{base}_sel",
        placeholder="Pick one or type a number",
    )

    if choice in (None, DEFAULT_LABEL, ""):
        return None
    try:
        value = int(str(choice).strip())
    except ValueError:
        st.warning(f"{spec.label}: '{choice}' is not a whole number - ignoring.")
        return None
    if spec.min is not None and value < spec.min:
        st.warning(f"{spec.label} must be at least {int(spec.min)} - ignoring.")
        return None
    if spec.max is not None and value > spec.max:
        st.warning(f"{spec.label} must be at most {int(spec.max)} - ignoring.")
        return None
    return value


def render_text(spec: cfg.ParamSpec, base: str, current: Any) -> list[str]:
    """Multi-pick list of stop sequences; new ones can be typed in."""
    chosen = list(current) if isinstance(current, list) else (
        [s.strip() for s in str(current).split(",") if s.strip()] if current else []
    )
    options = list(dict.fromkeys(spec.presets + chosen))

    return st.multiselect(
        spec.label,
        options,
        default=chosen,
        accept_new_options=True,
        help=spec.help,
        key=f"{base}_multi",
        placeholder="None - pick or type your own",
    )


# --- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.subheader("Model")

    provider = st.selectbox(
        "Provider", providers, format_func=lambda p: p.label,
        help="Only providers with an API key in .env are listed.",
    )

    model_options = provider.models
    model_name = st.selectbox(
        "Model", model_options,
        accept_new_options=True,
        help="Pick one, or type any model ID this provider accepts - "
             "published model lists go stale quickly.",
        placeholder="Pick a model or type an ID",
    )
    if not model_name:
        model_name = provider.models[0]

    if provider.notes:
        st.caption(provider.notes)

    note = cfg.fixed_sampling_note(provider, model_name)
    if note:
        st.info(note, icon="⚠️")

    mnote = cfg.model_note(provider, model_name)
    if mnote:
        st.caption(f"ℹ️ {mnote}")

    st.divider()

    head, icon = st.columns([3, 1], vertical_alignment="center")
    with head:
        st.subheader("Parameters")
    with icon:
        with st.popover("ℹ️", width="stretch"):
            st.markdown("#### Which model supports what")
            st.dataframe(cfg.support_rows(), hide_index=True, width="stretch")
            st.caption(
                "A control is hidden when the selected provider doesn't accept "
                "that parameter. Gemini 3.5 Flash Lite and Gemini 3.6 Flash also "
                "run at fixed sampling and ignore Temperature, Top K and Top P."
            )
            st.markdown("---")
            for spec in cfg.PARAM_SPECS:
                st.markdown(f"**{spec.label}**")
                st.markdown(spec.help)
                st.markdown("")

    settings = st.session_state.settings

    for spec in cfg.PARAM_SPECS:
        # The capability matrix at work: skip anything that wouldn't be applied.
        if not provider.supports(spec.key, model_name):
            continue

        base = f"w_{provider.key}_{spec.key}"
        current = settings.get(spec.key)

        if spec.kind == "float":
            settings[spec.key] = render_float(spec, base, current)
        elif spec.kind == "int":
            settings[spec.key] = render_int(spec, base, current)
        else:
            settings[spec.key] = render_text(spec, base, current)

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Reset params", width="stretch"):
            st.session_state.settings = cfg.default_settings()
            for spec in cfg.PARAM_SPECS:
                for suffix in ("_sld", "_num", "_sel", "_multi", "_pills"):
                    st.session_state.pop(f"w_{provider.key}_{spec.key}{suffix}", None)
            st.rerun()
    with col_b:
        if st.button("New chat", width="stretch", type="primary"):
            reset_thread(st.session_state.thread_id)
            st.session_state.history = []
            n = int(st.session_state.thread_id.split("-")[-1]) + 1
            st.session_state.thread_id = f"chat-{n}"
            st.rerun()

    with st.expander("Exactly what gets sent"):
        st.json(cfg.translate(provider, settings, model_name))


# --- main chat -------------------------------------------------------------
st.title("💬 Chat Assistant")
st.caption(f"{provider.label} · {model_name}")

for role, text, caption in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)
        if caption:
            st.caption(caption)

if prompt := st.chat_input("Ask me anything"):
    st.session_state.history.append(("user", prompt, None))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Rebuilt every message so the current settings apply, while the
            # shared checkpointer keeps the conversation intact.
            agent = build_agent(provider, model_name, settings)
            answer = st.write_stream(
                stream_reply(agent, prompt, st.session_state.thread_id)
            )
            # Built from what was actually sent, not from the widgets, so the
            # caption can't claim a value the model discarded.
            applied = cfg.translate(provider, settings, model_name)
            caption = f"{provider.label} · {model_name}"
            if applied.get("temperature") is not None:
                caption += f" · temp {applied['temperature']:g}"
            st.caption(caption)
        except Exception as err:
            answer = cfg.explain_error(err, provider, model_name)
            caption = None
            st.error(answer)

    st.session_state.history.append(("assistant", answer, caption))
