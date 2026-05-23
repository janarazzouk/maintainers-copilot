from __future__ import annotations

import os

import streamlit as st

from app.api_client import ApiError
from app.session import (
    get_api_client,
    is_admin,
    render_sidebar,
    require_login,
    show_api_error,
)

st.set_page_config(
    page_title="Admin Widgets | Maintainer's Copilot",
    page_icon="⚙️",
    layout="wide",
)

render_sidebar()

if not require_login():
    st.stop()

if not is_admin():
    st.error("Admin role is required to manage widget configurations.")
    st.stop()

client = get_api_client()
api_base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.title("⚙️ Admin widget configuration")
st.caption("Create and inspect embeddable widget configurations.")

tab_list, tab_create = st.tabs(["List widgets", "Create widget"])

with tab_list:
    st.subheader("Existing widgets")

    try:
        widgets = client.list_widgets()
    except ApiError as exc:
        show_api_error(exc)
        widgets = []

    if not widgets:
        st.info("No widgets created yet.")
    else:
        for widget in widgets:
            with st.container(border=True):
                st.write(f"**{widget['name']}**")
                st.code(widget["widget_id"])

                st.caption(
                    f"Active: `{widget['is_active']}` · Created: {widget['created_at']}"
                )

                public_config_url = f"{api_base_url}/widgets/{widget['widget_id']}/config"
                st.write("Public config URL:")
                st.code(public_config_url)

                with st.expander("Allowed origins"):
                    st.json(widget.get("allowed_origins", []))

                with st.expander("Theme"):
                    st.json(widget.get("theme", {}))

                with st.expander("Enabled tools"):
                    st.json(widget.get("enabled_tools", []))

                if st.button(
                    "Check public config",
                    key=f"check_widget_{widget['widget_id']}",
                ):
                    try:
                        public_config = client.get_public_widget_config(
                            widget_id=widget["widget_id"]
                        )
                        st.json(public_config)
                    except ApiError as exc:
                        show_api_error(exc)

with tab_create:
    st.subheader("Create widget")

    with st.form("create_widget_form"):
        name = st.text_input("Widget name", value="Local demo widget")

        allowed_origins_text = st.text_area(
            "Allowed origins",
            value="http://localhost:8090\nhttp://127.0.0.1:8090",
            help="One origin per line.",
        )

        primary_color = st.text_input("Primary color", value="#2563eb")
        position = st.selectbox(
            "Position",
            options=["bottom-right", "bottom-left"],
            index=0,
        )

        greeting = st.text_input(
            "Greeting",
            value="Ask me about resolved maintainer issues.",
        )

        enabled_tools = st.multiselect(
            "Enabled tools",
            options=[
                "rag_search",
                "classify_issue",
                "extract_entities",
                "summarize_thread",
                "write_memory",
            ],
            default=[
                "rag_search",
                "classify_issue",
                "extract_entities",
                "summarize_thread",
            ],
        )

        is_active = st.checkbox("Active", value=True)

        submitted = st.form_submit_button("Create widget", use_container_width=True)

    if submitted:
        allowed_origins = [
            line.strip()
            for line in allowed_origins_text.splitlines()
            if line.strip()
        ]

        try:
            widget = client.create_widget(
                name=name,
                allowed_origins=allowed_origins,
                theme={
                    "primaryColor": primary_color,
                    "position": position,
                },
                greeting=greeting,
                enabled_tools=enabled_tools,
                is_active=is_active,
            )

            st.success("Widget created.")
            st.json(widget)

            st.write("Public config URL:")
            st.code(f"{api_base_url}/widgets/{widget['widget_id']}/config")

        except ApiError as exc:
            show_api_error(exc)