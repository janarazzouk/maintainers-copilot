from __future__ import annotations

import streamlit as st

from app.api_client import ApiError
from app.session import get_api_client, render_sidebar, require_login, show_api_error

st.set_page_config(
    page_title="Chat | Maintainer's Copilot",
    page_icon="💬",
    layout="wide",
)

render_sidebar()

if not require_login():
    st.stop()

client = get_api_client()

st.title("💬 Chat")
st.caption("Ask the copilot what to check, what the issue likely means, or what action to take next.")

left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Conversations")

    if st.button("New conversation", use_container_width=True):
        st.session_state["active_conversation_id"] = None
        st.rerun()

    try:
        conversations = client.list_conversations()
    except ApiError as exc:
        show_api_error(exc)
        conversations = []

    if conversations:
        options = [None] + [item["id"] for item in conversations]

        def format_conversation(conversation_id: int | None) -> str:
            if conversation_id is None:
                return "New conversation"

            match = next(
                (item for item in conversations if item["id"] == conversation_id),
                None,
            )
            if not match:
                return f"Conversation {conversation_id}"

            return f"#{match['id']} — {match['title']}"

        current_id = st.session_state.get("active_conversation_id")
        if current_id not in options:
            current_id = None

        selected_id = st.selectbox(
            "Select conversation",
            options=options,
            index=options.index(current_id),
            format_func=format_conversation,
        )

        st.session_state["active_conversation_id"] = selected_id

        if selected_id is not None:
            if st.button("Delete selected conversation", use_container_width=True):
                try:
                    client.delete_conversation(conversation_id=selected_id)
                    st.session_state["active_conversation_id"] = None
                    st.success("Conversation deleted.")
                    st.rerun()
                except ApiError as exc:
                    show_api_error(exc)
    else:
        st.info("No conversations yet.")

with right_col:
    st.subheader("Messages")

    active_conversation_id = st.session_state.get("active_conversation_id")

    messages: list[dict] = []
    if active_conversation_id is not None:
        try:
            messages = client.list_messages(conversation_id=active_conversation_id)
        except ApiError as exc:
            show_api_error(exc)

    for message in messages:
        role = message["role"]
        content = message["content"]

        with st.chat_message(role if role in {"user", "assistant"} else "assistant"):
            st.write(content)

            snapshot_key = message.get("retrieval_snapshot_key")
            if snapshot_key:
                with st.expander("Developer evidence snapshot"):
                    st.code(snapshot_key)

    prompt = st.chat_input("Ask what the user should do next...")

    if prompt:
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = client.send_chat_message(
                        message=prompt,
                        conversation_id=active_conversation_id,
                    )
                except ApiError as exc:
                    show_api_error(exc)
                    st.stop()

            st.write(response["answer"])

            assistant_message = response.get("assistant_message", {})
            snapshot_key = assistant_message.get("retrieval_snapshot_key")
            if snapshot_key:
                with st.expander("Developer evidence snapshot"):
                    st.code(snapshot_key)

            st.session_state["active_conversation_id"] = response["conversation_id"]

        st.rerun()

    with st.expander("Example prompts"):
        st.code(
            "I upgraded to v1.2.0 and now app/main.py throws CONFIG_ERROR when calling load_config(). "
            "What is the likely cause and what files should I check?"
        )
        st.code(
            "Users say the CLI crashes after the latest release with TypeError in parse_config(). "
            "What should I ask them for?"
        )
        st.code(
            "Remember that for this project I prefer changed files only, not full project rewrites."
        )