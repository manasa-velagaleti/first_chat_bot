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

SYSTEM_PROMPT = """You are a helpful, concise assistant.

You can search the user's documents with the search_documents tool. Use it whenever a question might be answered by them; do not use it for chitchat or general knowledge.

Search again whenever the user asks about a different aspect of the documents. Passages you retrieved earlier only cover what was asked then, and treating them as the whole document leads you to say something is not mentioned when it simply was not searched for.

When you use a passage, cite it as [1] or [2]. Say plainly when the documents do not cover something rather than guessing.

You are not limited to the documents. Reason about what they say, compare it with what the user tells you, and combine the two - just make clear which part came from a document and which is your own reasoning."""

# One process-wide store of conversation history. It is deliberately NOT tied
# to any particular model, so agents can be rebuilt freely around it.
_CHECKPOINTER = MemorySaver()


def get_checkpointer() -> MemorySaver:
    return _CHECKPOINTER


def build_model(provider: Provider, model_name: str, settings: dict[str, Any]):
    """Instantiate the chat model with only the params this provider accepts."""
    kwargs = translate(provider, settings, model_name)
    return init_chat_model(f"{provider.prefix}:{model_name}", **kwargs)


def build_agent(provider: Provider, model_name: str,
                settings: dict[str, Any] | None = None,
                tools: list | None = None):
    """Build an agent. Cheap - no network call - so call it per message.

    To give the assistant abilities, define tools with the @tool decorator and
    pass them in tools=[...]; the agent decides when to call them.
    """
    return create_agent(
        model=build_model(provider, model_name, settings or default_settings()),
        tools=tools or [],
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
    """Yield the assistant's reply token by token, excluding tool output.

    stream_mode="messages" yields every message the graph produces, tool
    results included. Those have to be dropped: a tool returns the raw
    passages it found, and letting them through prefixes each answer with its
    own search results before the actual reply begins.

    The filter is on the graph node rather than the message class, because a
    streamed chunk is an AIMessageChunk either way - the node it came from is
    what distinguishes the model's own words from a tool's return value.
    """
    config = {"configurable": {"thread_id": thread_id}}
    for chunk, metadata in agent.stream(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        stream_mode="messages",
    ):
        if (metadata or {}).get("langgraph_node") == "tools":
            continue
        if type(chunk).__name__ == "ToolMessage":
            continue
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
