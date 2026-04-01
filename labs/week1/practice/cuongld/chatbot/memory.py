from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import streamlit as st


class Message(TypedDict):
    role: str
    content: str


@dataclass
class ChatMemory:
    key: str = "messages"

    def init(self) -> None:
        if self.key not in st.session_state:
            st.session_state[self.key] = []

    def all(self) -> list[Message]:
        self.init()
        return st.session_state[self.key]

    def add(self, role: str, content: str) -> None:
        self.init()
        st.session_state[self.key].append({"role": role, "content": content})

    def trim(self, max_history: int) -> None:
        self.init()
        if len(st.session_state[self.key]) > max_history:
            st.session_state[self.key] = st.session_state[self.key][-max_history:]

    def clear(self) -> None:
        st.session_state[self.key] = []
