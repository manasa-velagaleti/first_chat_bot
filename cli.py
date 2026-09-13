"""Terminal chat, using each provider's default settings.

For the model switcher and sampling controls, use the Streamlit app instead:
    streamlit run app.py
"""

from __future__ import annotations

import config as cfg
from chatbot import build_agent, stream_reply

THREAD_ID = "cli-session"


def main() -> None:
    providers = cfg.available_providers()
    if not providers:
        print(
            "No API keys found. Add one of "
            + ", ".join(p.env_key for p in cfg.PROVIDERS.values())
            + " to your .env file."
        )
        return

    provider = providers[0]
    model_name = provider.models[0]
    agent = build_agent(provider, model_name)

    print(f"Chat ready ({provider.label} · {model_name}). Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            return

        print("Bot: ", end="", flush=True)
        try:
            for token in stream_reply(agent, user_input, THREAD_ID):
                print(token, end="", flush=True)
        except Exception as err:  # keep the chat alive on a bad turn
            print(f"\n[error: {err}]")
        print("\n")


if __name__ == "__main__":
    main()
