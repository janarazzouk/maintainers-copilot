from __future__ import annotations

import streamlit as st

from app.api_client import ApiError
from app.session import get_api_client, render_sidebar, require_login, show_api_error

st.set_page_config(
    page_title="Memory | Maintainer's Copilot",
    page_icon="🧠",
    layout="wide",
)

render_sidebar()

if not require_login():
    st.stop()

client = get_api_client()

st.title("🧠 Long-term memory")
st.caption("Inspect, search, create, and delete the authenticated user's long-term memory.")

tab_list, tab_search, tab_create = st.tabs(["List memory", "Search memory", "Create memory"])

with tab_list:
    st.subheader("Stored memories")

    try:
        memories = client.list_memories()
    except ApiError as exc:
        show_api_error(exc)
        memories = []

    if not memories:
        st.info("No memories stored yet.")
    else:
        for memory in memories:
            with st.container(border=True):
                st.write(f"**Memory #{memory['id']}**")
                st.write(memory["redacted_content"])
                st.caption(f"Type: `{memory['memory_type']}` · Created: {memory['created_at']}")

                metadata = memory.get("memory_metadata") or {}
                if metadata:
                    with st.expander("Metadata"):
                        st.json(metadata)

                if st.button(
                    "Delete",
                    key=f"delete_memory_{memory['id']}",
                    type="secondary",
                ):
                    try:
                        client.delete_memory(memory_id=memory["id"])
                        st.success(f"Deleted memory #{memory['id']}.")
                        st.rerun()
                    except ApiError as exc:
                        show_api_error(exc)

with tab_search:
    st.subheader("Search memories")

    query = st.text_input("Search query", value="How does the user prefer code fixes?")
    limit = st.slider("Limit", min_value=1, max_value=20, value=5)

    if st.button("Search memory", use_container_width=True):
        try:
            results = client.search_memory(query=query, limit=limit)
        except ApiError as exc:
            show_api_error(exc)
            results = []

        if not results:
            st.info("No matching memories found.")
        else:
            for result in results:
                with st.container(border=True):
                    st.write(f"**Memory #{result['id']}**")
                    st.write(result["content"])
                    st.caption(f"Score: `{result['score']}` · Type: `{result['memory_type']}`")

                    metadata = result.get("metadata") or {}
                    if metadata:
                        with st.expander("Metadata"):
                            st.json(metadata)

with tab_create:
    st.subheader("Create memory manually")

    with st.form("create_memory_form"):
        memory_type = st.selectbox(
            "Memory type",
            options=["semantic", "episodic", "procedural"],
            index=0,
        )
        content = st.text_area(
            "Content",
            height=120,
            placeholder="The user prefers changed-files-only patches, not full project rewrites.",
        )
        reason = st.text_input(
            "Reason",
            placeholder="Useful for future project code fixes.",
        )

        submitted = st.form_submit_button("Create memory", use_container_width=True)

    if submitted:
        try:
            memory = client.create_memory(
                memory_type=memory_type,
                content=content,
                reason=reason or None,
            )
            st.success(f"Memory saved as #{memory['id']}.")
            st.json(memory)
        except ApiError as exc:
            show_api_error(exc)

st.divider()

st.info(
    "Memory is written explicitly through the /memory endpoint or by the chatbot's write_memory tool. "
    "Secrets are redacted before storage."
)