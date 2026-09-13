"""Agent construction and streaming.

The important idea here is that the agent and the conversation history have
different lifetimes. create_agent() bakes the model in, so changing any
sampling parameter means building a new agent - but the history must survive
that. Keeping the checkpointer separate (see get_checkpointer) is what makes
"move the temperature slider mid-chat" not wipe the conversation.
"""

from __future__ import annotations

from typing import Any, Iterator

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from config import Provider, default_settings, translate

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. "
    "Answer directly, and say so plainly when you don't know something."
)

# One process-wide store of conversation history. It is deliberately NOT tied
# to any particular model, so agents can be rebuilt freely around it.
_CHECKPOINTER = MemorySaver()


def get_checkpointer() -> MemorySaver:
    return _CHECKPOINTER


def build_model(provider: Provider, model_name: str, settings: dict[str, Any]):
    """Instantiate the chat model with only the params this provider accepts."""
    kwargs = translate(provider, settings, model_name)
    return init_chat_model(f"{provider.prefix}:{model_name}", **kwargs)


def build_agent(provider: Provider, model_name: str, settings: dict[str, Any] | None = None):
    """Build an agent. Cheap - no network call - so call it per message.

    To give the assistant abilities, define tools with the @tool decorator and
    pass them in tools=[...]; the agent decides when to call them.
    """
    return create_agent(
        model=build_model(provider, model_name, settings or default_settings()),
        tools=[],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=get_checkpointer(),
    )


def chunk_text(chunk) -> str:
    """Pull plain text out of a streamed chunk.

    Providers differ: some stream a plain string, others stream a list of
    content blocks (text, reasoning, tool calls). Only text is displayed.
    """
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def stream_reply(agent, message: str, thread_id: str = "default") -> Iterator[str]:
    """Yield the reply token by token so the UI can render it as it arrives."""
    config = {"configurable": {"thread_id": thread_id}}
    for chunk, _metadata in agent.stream(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        stream_mode="messages",
    ):
        text = chunk_text(chunk)
        if text:
            yield text


def reply(agent, message: str, thread_id: str = "default") -> str:
    """Non-streaming variant: return the whole reply as one string."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )
    return result["messages"][-1].text


def reset_thread(thread_id: str) -> None:
    """Forget one conversation, leaving other threads untouched."""
    try:
        _CHECKPOINTER.delete_thread(thread_id)
    except Exception:
        # Older checkpointer versions lack delete_thread; the caller falls back
        # to switching to a fresh thread_id, which is equivalent from the UI.
        pass
