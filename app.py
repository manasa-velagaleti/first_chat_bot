"""Multi-model chat assistant.  Run:  streamlit run app.py"""

from __future__ import annotations

from typing import Any

import streamlit as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "embeddings_app"))

import core                          # noqa: E402  SUPPORTED types
import sessions                      # noqa: E402  shared with rag_web
import config as cfg
import rag_tools
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

# Which Pinecone namespace the assistant searches. chars700 suits a
# PDF-heavy corpus; paragraph chunking fragments PDF pages badly.
DOC_NAMESPACE = "chars700"

# --- session state ---------------------------------------------------------
if "settings" not in st.session_state:
    st.session_state.settings = cfg.default_settings()
if "history" not in st.session_state:
    st.session_state.history = []          # [(role, text, caption, sources)]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "chat-1"
if "use_docs" not in st.session_state:
    st.session_state.use_docs = True
if "upload_round" not in st.session_state:
    st.session_state.upload_round = 0
if "sid" not in st.session_state:
    # One scratch corpus per browser session. Uploads live in an OS temp
    # folder and are discarded when it is cleared.
    st.session_state.sid = sessions.start().id


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
    st.subheader("Documents")

    can_search = provider.supports_tools(model_name)
    if can_search:
        st.session_state.use_docs = st.toggle(
            "Search my documents",
            value=st.session_state.use_docs,
            help="When on, the assistant searches your documents if a question "
                 "looks like it relates to them. Ordinary chat is unaffected - "
                 "it only searches when it judges it useful.",
        )

        sess = sessions.get(st.session_state.sid)

        upload = st.file_uploader(
            "Add a document",
            type=[e.lstrip(".") for e in core.SUPPORTED],
            help="Read and embedded straight away, then searchable. Kept in a "
                 "temporary folder outside this project and deleted when you "
                 "clear the session.",
            key=f"up_{st.session_state.upload_round}",
        )
        if upload is not None and sess is not None:
            with st.spinner(f"Reading and embedding {upload.name}…"):
                res = sessions.add_document(sess, upload.name, upload.getvalue())
            if res.get("error"):
                st.error(res["error"])
            else:
                st.success(f"{res['name']} — {res['chunks']} chunks")
                rag_tools.drop_store(sess.namespace)   # reopen on new vectors
            # Reset the widget so the same file can be re-added after a clear,
            # and so this block does not re-run on the next interaction.
            st.session_state.upload_round += 1
            st.rerun()

        if sess and sess.files:
            st.caption("**This session**")
            for f in sess.files:
                st.caption(f"· {f['name']} — {f['chunks']} chunks")
            if st.button("Clear session documents", width="stretch"):
                sessions.clear(sess.id)
                rag_tools.drop_store(sess.namespace)
                st.session_state.sid = sessions.start().id
                st.rerun()
            st.caption("Searching your uploads only.")
        else:
            corpus = rag_tools.describe_corpus(DOC_NAMESPACE)
            st.caption(f"Searching saved documents: {corpus}" if corpus
                       else "No documents indexed yet.")
    else:
        st.info(f"`{model_name}` can't use tools, so it can't search "
                f"documents. Chat works normally.", icon="⚠️")

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

def render_sources(sources):
    """The passages the assistant actually retrieved for one answer."""
    if not sources:
        return
    with st.expander(f"📄 {len(sources)} passage"
                     f"{'' if len(sources) == 1 else 's'} used"):
        for src in sources:
            where = f"page {src['page']}" if src.get("page") is not None else (
                f"paragraph {src['paragraph']}"
                if src.get("paragraph") is not None else "")
            st.caption(f"**[{src['n']}]** {src['file']}"
                       + (f" · {where}" if where else "")
                       + f" · score {src['score']}")
            st.markdown(f"> {' '.join(src['text'].split())}")


for role, text, caption, sources in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)
        if sources:
            render_sources(sources)
        if caption:
            st.caption(caption)

if prompt := st.chat_input("Ask me anything"):
    st.session_state.history.append(("user", prompt, None, None))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        sources = None
        try:
            thread = st.session_state.thread_id
            # Only hand over the tool when the model can use it and the user
            # wants it; otherwise this is an ordinary chat agent.
            tools = None
            if can_search and st.session_state.use_docs:
                rag_tools.clear_sources(thread)     # so stale hits aren't shown
                # Uploads take precedence: if you put a document into this
                # session, that is what you mean by "my documents". Falling
                # back to the saved corpus would answer from files you did not
                # just hand over - the failure that is hardest to spot.
                sess = sessions.get(st.session_state.sid)
                ns = sess.namespace if (sess and sess.files) else DOC_NAMESPACE
                tools = [rag_tools.make_search_tool(thread, ns, k=4)]

            # Rebuilt every message so the current settings apply, while the
            # shared checkpointer keeps the conversation intact.
            agent = build_agent(provider, model_name, settings, tools=tools)
            answer = st.write_stream(
                stream_reply(agent, prompt, thread)
            )

            # Populated by the tool only if the model chose to search.
            sources = rag_tools.last_sources(thread) or None
            if sources:
                render_sources(sources)

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

    st.session_state.history.append(("assistant", answer, caption, sources))
