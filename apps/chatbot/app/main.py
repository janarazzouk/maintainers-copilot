from __future__ import annotations

import streamlit as st

from app.api_client import ApiError
from app.session import (
    get_user,
    init_session,
    is_logged_in,
    login,
    render_sidebar,
    show_api_error,
)

st.set_page_config(
    page_title="Maintainer's Copilot",
    page_icon="🛠️",
    layout="wide",
)

init_session()
render_sidebar()

st.title("🛠️ Maintainer's Copilot")
st.caption("Internal chatbot and admin UI for issue triage, RAG, memory, and widget configuration.")

if is_logged_in():
    user = get_user()

    st.success(f"You are logged in as {user.get('email')}.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.page_link("pages/chat.py", label="Open chatbot", icon="💬")

    with col2:
        st.page_link("pages/memory.py", label="Inspect memory", icon="🧠")

    with col3:
        if user.get("role") == "admin":
            st.page_link("pages/admin_widgets.py", label="Admin widgets", icon="⚙️")
        else:
            st.info("Admin widget settings are admin-only.")

    st.divider()

    st.subheader("What this UI demonstrates")
    st.write(
        """
        This Streamlit app talks to the FastAPI backend. The backend handles JWT auth,
        Groq tool-calling, classifier/NER/RAG tools, Redis short-term memory,
        Postgres long-term memory, MinIO RAG snapshots, and audit logs.
        """
    )

else:
    st.subheader("Login")

    with st.form("login_form"):
        email = st.text_input("Email", value="admin@example.com")
        password = st.text_input("Password", value="password123", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

    if submitted:
        try:
            login(email=email, password=password)
            st.success("Logged in successfully.")
            st.rerun()
        except ApiError as exc:
            show_api_error(exc)

    st.info("Use the admin user you already created through the API.")