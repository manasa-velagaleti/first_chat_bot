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
import evaluation as ev
import rag_tools
import voice
from streamlit_mic_recorder import speech_to_text
from chatbot import build_agent, stream_reply

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
# --- chats ------------------------------------------------------------------
# Several conversations per run. Each owns its message history, its LangGraph
# thread (so memory stays separate), and its own scratch corpus for documents
# attached to that chat alone. Nothing is persisted - this lasts the session.
if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.chat_order = []
    st.session_state.current = None


def new_chat() -> str:
    n = len(st.session_state.chat_order) + 1
    cid = f"chat-{n}-{len(st.session_state.chats)}"
    st.session_state.chats[cid] = {
        "title": f"Chat {n}",
        "history": [],                 # [(role, text, caption, sources)]
        "thread": cid,                 # LangGraph thread id
        "sid": sessions.start().id,    # this chat's own documents
    }
    st.session_state.chat_order.append(cid)
    st.session_state.current = cid
    return cid


def chat() -> dict:
    if st.session_state.current not in st.session_state.chats:
        new_chat()
    return st.session_state.chats[st.session_state.current]


if not st.session_state.chats:
    new_chat()
if "use_docs" not in st.session_state:
    st.session_state.use_docs = True
if "spoken" not in st.session_state:
    st.session_state.spoken = {}        # (chat, turn) -> mp3 bytes
if "voice_round" not in st.session_state:
    st.session_state.voice_round = 0
if "verdicts" not in st.session_state:
    st.session_state.verdicts = {}      # (chat, turn) -> Verdict
if "upload_round" not in st.session_state:
    st.session_state.upload_round = 0
if "shared_sid" not in st.session_state:
    # Documents uploaded in the sidebar: visible to every chat in this run.
    # Per-chat documents live in each chat's own session instead.
    st.session_state.shared_sid = sessions.start().id


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
cur = chat()
chat_sess = sessions.get(cur["sid"])

# What this chat can search: documents shared across the app, plus any
# attached to this chat alone. A chat with its own files still sees the
# shared ones - attaching a document narrows nothing.
search_ns = []
shared_sess = sessions.get(st.session_state.shared_sid)
if shared_sess and shared_sess.files:
    search_ns.append(shared_sess.namespace)
if chat_sess and chat_sess.files:
    search_ns.append(chat_sess.namespace)
if not search_ns:
    search_ns = [DOC_NAMESPACE]        # fall back to the saved corpus

with st.sidebar:
    st.subheader("Chats")
    if st.button("➕ New chat", width="stretch", type="primary"):
        new_chat()
        st.rerun()

    for cid in st.session_state.chat_order:
        c = st.session_state.chats[cid]
        docs = len(sessions.get(c["sid"]).files) if sessions.get(c["sid"]) else 0
        label = c["title"] + (f"  📎{docs}" if docs else "")
        if st.button(label, key=f"go_{cid}", width="stretch",
                     type="secondary",
                     disabled=cid == st.session_state.current):
            st.session_state.current = cid
            st.rerun()

    st.divider()
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

        shared = sessions.get(st.session_state.shared_sid)

        upload = st.file_uploader(
            "Add a document for every chat",
            type=[e.lstrip(".") for e in core.SUPPORTED],
            help="Embedded on upload and searchable from any chat. Kept in a "
                 "temporary folder outside this project, and gone when you "
                 "clear it or close the app. To attach a document to one chat "
                 "only, use the paperclip in the chat box instead.",
            key=f"up_{st.session_state.upload_round}",
        )
        if upload is not None and shared is not None:
            with st.spinner(f"Reading and embedding {upload.name}…"):
                res = sessions.add_document(shared, upload.name, upload.getvalue())
            if res.get("error"):
                st.error(res["error"])
            else:
                st.success(
                    f"**{res['name']}** — {res['characters']:,} characters "
                    f"→ **{res['chunks']} chunks** → embedded with "
                    f"{core.EMBED_MODEL}"
                )
                rag_tools.drop_store(shared.namespace)  # reopen on new vectors
            # Reset the widget so the same file can be re-added after a clear,
            # and so this block does not re-run on the next interaction.
            st.session_state.upload_round += 1
            st.rerun()

        if shared and shared.files:
            st.caption("**Shared with every chat**")
            for f in shared.files:
                st.caption(f"· {f['name']} — {f['chunks']} chunks")
            if st.button("Clear shared documents", width="stretch"):
                sessions.clear(shared.id)
                rag_tools.drop_store(shared.namespace)
                st.session_state.shared_sid = sessions.start().id
                st.rerun()
        else:
            corpus = rag_tools.describe_corpus(DOC_NAMESPACE)
            st.caption(f"Also searching saved documents: {corpus}" if corpus
                       else "No saved documents indexed.")
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

    # "New chat" lives at the top of the sidebar with the chat list now.
    st.divider()
    st.subheader("Voice")
    if voice.available():
        st.session_state.tts_voice = st.selectbox(
            "Voice for reading answers", voice.VOICES,
            help="Used by the Read aloud button under each answer.",
        )
        st.session_state.auto_speak = st.toggle(
            "Speak answers automatically",
            value=st.session_state.get("auto_speak", True),
            help="Read each new answer aloud as it arrives. Turn off to use "
                 "the Read aloud button only.",
        )
        st.caption("The microphone is next to the chat box.")
    else:
        st.caption("Voice needs OPENAI_API_KEY, even when another provider "
                   "is answering.")

    st.divider()
    st.subheader("Answer evaluation")
    st.caption("A second model checks whether an answer is actually supported "
               "by the passages it cited. Every model can judge — it needs "
               "no tools — including the three that cannot search.")

    judge_provider = st.selectbox(
        "Judge provider", providers, format_func=lambda p: p.label,
        index=next((i for i, p in enumerate(providers)
                    if p.key != provider.key), 0),
        help="Best kept different from the chat model. Two models from the "
             "same family share training, so they tend to share mistakes - "
             "and a model marking its own work mostly agrees with itself.",
    )
    judge_model = st.selectbox("Judge model", judge_provider.models,
                               key="judge_model")
    st.session_state.judge = (judge_provider, judge_model)
    if judge_provider.key == provider.key:
        st.caption("⚠️ Judging with the same provider that answered.")

    st.divider()
    with st.expander("ℹ️ Which models can search documents?"):
        st.caption("Document search needs tool calling. Most models support "
                   "it; these do not, and run as plain chat instead.")
        rows = []
        for prov in cfg.PROVIDERS.values():
            for mdl in prov.models:
                rows.append({
                    "Provider": prov.label,
                    "Model": mdl,
                    "Can search": "Yes" if prov.supports_tools(mdl) else "No",
                })
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("A model can also be temporarily unavailable through free-"
                   "tier quota - that shows as a rate-limit error, not as a "
                   "missing feature. Switching model fixes it.")

    with st.expander("📊 What is stored"):
        if st.button("Check index", key="idx_stats"):
            st.session_state.idx = rag_tools.index_stats(search_ns)
        info = st.session_state.get("idx") if "idx" in st.session_state else None
        if info:
            st.caption(f"**Embedding model**  {info['embed_model']}")
            st.caption(f"**Dimensions**  {info['dimensions']} per chunk")
            st.caption(f"**Distance metric**  {info['metric']}")
            st.caption(f"**Vectors stored**  {info['vectors']:,}")
            for ns, n in info["per_namespace"].items():
                label = ("this chat" if chat_sess and ns == chat_sess.namespace
                         else "shared" if shared_sess and ns == shared_sess.namespace
                         else "saved documents")
                st.caption(f"· {label}: {n:,}")

    st.divider()
    with st.expander("🔍 Browse chunks"):
        st.caption("Every chunk the assistant can search — not just the ones "
                   "a question retrieved. This is how your documents were "
                   "actually cut up.")
        if st.button("Load chunks", key="load_chunks"):
            st.session_state.chunks = rag_tools.all_chunks(search_ns)

        chunks = st.session_state.get("chunks")
        if chunks:
            files = sorted({c["file"] for c in chunks})
            pick = st.selectbox("Document", ["All"] + files, key="chunk_file")
            shown = [c for c in chunks if pick == "All" or c["file"] == pick]
            st.caption(f"{len(shown)} chunk{'' if len(shown) == 1 else 's'}")

            per = 10
            pages = max(1, (len(shown) + per - 1) // per)
            page = st.number_input("Page", 1, pages, 1, key="chunk_page") if pages > 1 else 1
            for c in shown[(page - 1) * per: page * per]:
                where = f" · p{c['page']}" if c["page"] is not None else ""
                st.caption(f"**{c['file']}**{where} · {len(c['text'])} chars")
                st.text(" ".join(c["text"].split())[:400])

    st.divider()
    if st.button("Reset params", width="stretch"):
        st.session_state.settings = cfg.default_settings()
        for spec in cfg.PARAM_SPECS:
            for suffix in ("_sld", "_num", "_sel", "_multi", "_pills"):
                st.session_state.pop(f"w_{provider.key}_{spec.key}{suffix}", None)
        st.rerun()

    with st.expander("Exactly what gets sent"):
        st.json(cfg.translate(provider, settings, model_name))


# --- main chat -------------------------------------------------------------

def render_speak(key: str, answer: str):
    """Read one answer aloud, on request rather than automatically.

    On request because synthesis costs a few seconds and a few cents, and an
    answer that is being skimmed does not need saying out loud.
    """
    if not voice.available():
        return

    audio = st.session_state.spoken.get(key)
    if audio is not None:
        # Already spoken - offer a replay rather than synthesising again.
        st.audio(audio, format="audio/mp3")
        return

    if st.button("🔊 Read aloud", key=f"sp_{key}",
                 help="Speak this answer using OpenAI text-to-speech."):
        with st.spinner("Generating speech…"):
            res = voice.speak(answer, st.session_state.get("tts_voice", "alloy"))
        if res.get("error"):
            st.error(res["error"])
        else:
            st.session_state.spoken[key] = res["audio"]
            if res.get("truncated"):
                st.caption("Long answer — reading the opening only.")
            st.rerun()


def render_evaluation(key: str, question: str, sources: list, answer: str,
                      searched: bool = True):
    """A judge button under one answer, and its verdict once it has run.

    When there is nothing to judge, say so. An answer with no cited passages
    has no reference text, and silently omitting the button leaves the user
    looking for a control that was never going to appear.
    """
    if not sources:
        if searched:
            st.caption("⚖️ Evaluation needs cited passages — this answer "
                       "did not come from the documents.")
        return

    verdict = st.session_state.verdicts.get(key)

    if st.button("⚖️ Evaluate this answer", key=f"ev_{key}",
                 help="Ask a second model whether this answer is supported "
                      "by the passages it cited."):
        jp, jm = st.session_state.judge
        with st.spinner(f"{jm} is checking the answer…"):
            st.session_state.verdicts[key] = ev.judge_answer(
                question, sources, answer, jp, jm)
        st.rerun()

    if verdict:
        icon = {"correct": "✅", "incorrect": "❌"}.get(
            verdict.verdict, "❓")
        with st.popover(f"{icon} Verdict: {verdict.verdict}"):
            st.markdown(f"### {icon} {verdict.verdict}")
            st.caption(f"Judged by {verdict.judge}")
            if verdict.error:
                st.warning(verdict.error)
            else:
                st.caption(f"The judge replied: `{verdict.raw}`")
            st.markdown("---")
            st.caption(
                "A second model was shown the question, the passages cited "
                "above, and the answer — then asked whether the answer is "
                "supported. \"incorrect\" also covers answers that are only "
                "partially right."
            )


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


st.title("💬 " + cur["title"])
bits = [f"{provider.label} · {model_name}"]
if chat_sess and chat_sess.files:
    bits.append("📎 " + ", ".join(f["name"] for f in chat_sess.files))
st.caption(" · ".join(bits))

if chat_sess and chat_sess.files:
    with st.expander(f"📎 {len(chat_sess.files)} document"
                     f"{'' if len(chat_sess.files) == 1 else 's'} in this chat"):
        for f in chat_sess.files:
            st.caption(f"· {f['name']} — {f['chunks']} chunks, "
                       f"{f['characters']:,} characters")
        if st.button("Remove this chat's documents", key="clr_chat"):
            sessions.clear(chat_sess.id)
            rag_tools.drop_store(chat_sess.namespace)
            cur["sid"] = sessions.start().id
            st.rerun()

for turn, (role, text, caption, sources) in enumerate(cur["history"]):
    with st.chat_message(role):
        st.markdown(text)
        if sources:
            render_sources(sources)
        if role == "assistant":
            # The question is the user turn immediately before this answer.
            asked = cur["history"][turn - 1][1] if turn else ""
            render_speak(f"{st.session_state.current}-{turn}", text)
            render_evaluation(f"{st.session_state.current}-{turn}",
                              asked, sources, text)
        if caption:
            st.caption(caption)

# A transcription waiting to be sent: shown in an editable box so it can be
# corrected before it goes. st.chat_input cannot be filled programmatically,
# so "put it in the prompt" means putting it somewhere editable next to it.
pending = st.session_state.get("pending_voice")
if pending:
    st.caption("🎤 Heard — edit if needed, then send:")
    edit_col, send_col, drop_col = st.columns([6, 1, 1])
    with edit_col:
        edited = st.text_input("Transcribed question", value=pending,
                               label_visibility="collapsed",
                               key=f"vt_{st.session_state.voice_round}")
    with send_col:
        if st.button("Send", width="stretch", type="primary"):
            st.session_state.voice_prompt = edited
            st.session_state.pop("pending_voice", None)
            st.session_state.voice_round += 1
            st.rerun()
    with drop_col:
        if st.button("Clear", width="stretch"):
            st.session_state.pop("pending_voice", None)
            st.session_state.voice_round += 1
            st.rerun()


# A small round mic button sitting level with the chat box's send arrow.
# The component renders a plain button, so the icon is drawn with a Material
# glyph and the label text is hidden - a record dot reads as a toggle, not as
# a microphone.
st.markdown("""
<link rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded"/>
<style>
  div[data-testid="stCustomComponentV1"] iframe { height: 46px !important; }
  div[data-testid="stCustomComponentV1"] { margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# Microphone, immediately right of the chat box.
#
# This uses the browser's own speech recognition rather than recording audio
# and sending it away to be transcribed. st.audio_input produced empty
# recordings here, and the round trip it needs - capture, upload, Whisper,
# back - has several places to fail silently. The browser returns text
# directly and needs no API key.
if voice.available() or True:
    box_col, mic_col = st.columns([14, 1], vertical_alignment="bottom",
                                  gap="small")
    with box_col:
        entry = st.chat_input(
            "Ask me anything — attach a document with the paperclip",
            accept_file=True,
            file_type=[e.lstrip(".") for e in core.SUPPORTED],
        )
    with mic_col:
        heard = speech_to_text(
            start_prompt="●",          # record
            stop_prompt="■",           # stop
            just_once=True,
            use_container_width=True,
            language="en",
            key="stt",
        )
        if heard:
            st.session_state.pending_voice = heard
            st.rerun()

# A transcribed recording enters here as though it had been typed, so voice
# and keyboard take exactly the same path through the rest of the app.
spoken_prompt = st.session_state.pop("voice_prompt", None)

if entry or spoken_prompt:
    if spoken_prompt:
        prompt, attached = spoken_prompt, []
        st.caption(f"🎤 Heard: {spoken_prompt}")
    else:
        # accept_file makes chat_input return an object with .text and .files
        prompt = (entry.text or "").strip() if hasattr(entry, "text") else str(entry)
        attached = list(getattr(entry, "files", []) or [])

    # Attachments belong to this chat only, and are embedded before the
    # question is answered so the first reply can already use them.
    if attached and chat_sess is not None:
        for up in attached:
            with st.spinner(f"Reading and embedding {up.name}…"):
                res = sessions.add_document(chat_sess, up.name, up.getvalue())
            if res.get("error"):
                st.error(f"{up.name}: {res['error']}")
            else:
                st.success(
                    f"Attached **{res['name']}** — {res['characters']:,} "
                    f"characters → **{res['chunks']} chunks** → embedded "
                    f"with `{core.EMBED_MODEL}`"
                )
                rag_tools.drop_store(chat_sess.namespace)
        if chat_sess.namespace not in search_ns:
            search_ns.append(chat_sess.namespace)
        if not prompt:
            st.rerun()      # attachment only, nothing to answer yet

    if prompt:
        cur["history"].append(("user", prompt, None, None))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            sources = None
            try:
                thread = cur["thread"]
                tools = None
                if can_search and st.session_state.use_docs:
                    rag_tools.clear_sources(thread)   # don't show stale hits
                    tools = [rag_tools.make_search_tool(thread, search_ns, k=4)]

                # Rebuilt every message so current settings apply, while the
                # shared checkpointer keeps this thread's memory intact.
                agent = build_agent(provider, model_name, settings, tools=tools)
                answer = st.write_stream(stream_reply(agent, prompt, thread))

                # Populated by the tool only if the model chose to search.
                sources = rag_tools.last_sources(thread) or None
                if sources:
                    render_sources(sources)
                turn_key = f"{st.session_state.current}-{len(cur['history'])}"

                # Speak it straight away when asked to. Done here rather than
                # on the next rerun so the audio arrives with the answer
                # instead of a beat later.
                if (voice.available()
                        and st.session_state.get("auto_speak", True)
                        and turn_key not in st.session_state.spoken):
                    with st.spinner("Speaking…"):
                        said = voice.speak(
                            answer, st.session_state.get("tts_voice", "alloy"))
                    if said.get("audio"):
                        st.session_state.spoken[turn_key] = said["audio"]
                        st.audio(said["audio"], format="audio/mp3",
                                 autoplay=True)
                        if said.get("truncated"):
                            st.caption("Long answer — reading the opening only.")

                render_speak(turn_key, answer)
                render_evaluation(turn_key, prompt, sources, answer)

                # Built from what was actually sent, not from the widgets, so
                # the caption can't claim a value the model discarded.
                applied = cfg.translate(provider, settings, model_name)
                caption = f"{provider.label} · {model_name}"
                if applied.get("temperature") is not None:
                    caption += f" · temp {applied['temperature']:g}"
                st.caption(caption)
            except Exception as err:
                answer = cfg.explain_error(err, provider, model_name)
                caption = None
                st.error(answer)

        cur["history"].append(("assistant", answer, caption, sources))

        # Name the chat after its first question, so the list is readable.
        if cur["title"].startswith("Chat ") and len(cur["history"]) <= 2:
            cur["title"] = (prompt[:28] + "…") if len(prompt) > 28 else prompt
            st.rerun()
