from __future__ import annotations

import streamlit as st

from app.api_client import APIClient, ApiError


def init_session() -> None:
    st.session_state.setdefault("token", None)
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("active_conversation_id", None)


def get_token() -> str | None:
    init_session()
    return st.session_state.get("token")


def get_user() -> dict | None:
    init_session()
    return st.session_state.get("user")


def is_logged_in() -> bool:
    return bool(get_token() and get_user())


def is_admin() -> bool:
    user = get_user()
    return bool(user and user.get("role") == "admin")


def get_api_client() -> APIClient:
    return APIClient(token=get_token())


def login(*, email: str, password: str) -> None:
    client = APIClient()

    token_response = client.login(email=email, password=password)
    token = token_response["access_token"]

    authed_client = APIClient(token=token)
    user = authed_client.me()

    st.session_state["token"] = token
    st.session_state["user"] = user


def logout() -> None:
    keys_to_clear = [
        "token",
        "user",
        "active_conversation_id",
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)


def require_login() -> bool:
    init_session()

    if is_logged_in():
        return True

    st.warning("Please log in from the main page first.")
    st.page_link("main.py", label="Go to login", icon="🔐")
    return False


def render_sidebar() -> None:
    init_session()

    with st.sidebar:
        st.title("Maintainer's Copilot")

        user = get_user()
        if user:
            st.caption("Signed in as")
            st.write(f"**{user.get('email')}**")
            st.write(f"Role: `{user.get('role')}`")

            if st.button("Logout", use_container_width=True):
                logout()
                st.rerun()

            st.divider()

            st.page_link("main.py", label="Home", icon="🏠")
            st.page_link("pages/chat.py", label="Chat", icon="💬")
            st.page_link("pages/memory.py", label="Memory", icon="🧠")

            if user.get("role") == "admin":
                st.page_link("pages/admin_widgets.py", label="Admin widgets", icon="⚙️")
        else:
            st.caption("Not signed in")


def show_api_error(error: ApiError) -> None:
    if error.status_code:
        st.error(f"API error {error.status_code}: {error}")
    else:
        st.error(f"API error: {error}")