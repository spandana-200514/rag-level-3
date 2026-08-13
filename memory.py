import streamlit as st


def initialize_memory():
    """Initialize chat history."""

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def add_message(role, message):
    """Store a message in chat history."""

    st.session_state.chat_history.append({
        "role": role,
        "content": message
    })


def get_chat_history():
    """Return previous conversation."""

    return st.session_state.chat_history


def clear_memory():
    """Clear conversation memory."""

    st.session_state.chat_history = []