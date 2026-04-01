import streamlit as st
from time import perf_counter

from agent import AgentReply, ChatbotAgent
from config import get_settings
from memory import ChatMemory
from pdf_utils import plan_markdown_to_pdf_bytes
import re


def _assistant_message_for_memory(reply: AgentReply) -> str:
    if not reply.next_actions:
        return reply.text
    lines = "\n".join(f"- {a}" for a in reply.next_actions)
    return f"{reply.text}\n\n---\n**Helpful next steps**\n{lines}"


_ITINERARY_LIKELY_RE = re.compile(
    r"(^\s*#\s+.*itinerar.*$)|(^\s*##\s*(day|ngày)\s*\d+)|(\b(day|ngày)\s*\d+\b)",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_latest_itinerary_from_memory(history: list[dict]) -> str | None:
    for msg in reversed(history or []):
        if msg.get("role") != "assistant":
            continue
        text = (msg.get("content") or "").strip()
        if not text:
            continue
        if _ITINERARY_LIKELY_RE.search(text):
            return text
    return None


def _is_itinerary_text(text: str | None) -> bool:
    return bool(text and _ITINERARY_LIKELY_RE.search(text))


def main() -> None:
    settings = get_settings()
    memory = ChatMemory()
    if "agent" not in st.session_state:
        st.session_state["agent"] = ChatbotAgent(system_prompt=settings.system_prompt)
    agent: ChatbotAgent = st.session_state["agent"]

    st.set_page_config(page_title=settings.chatbot_title, page_icon="🤖", layout="centered")
    st.title(settings.chatbot_title)
    st.caption(
        "Collect trip length (1–7 days), food, then places. "
        "When the review appears, reply once with **confirm** or **ok** to generate the itinerary."
    )

    with st.sidebar:
        st.subheader("Session Tools")
        st.caption("Conversation memory: SQLite checkpoints")
        st.caption(f"Thread ID: {agent.thread_id}")
        if st.button("Clear chat", use_container_width=True):
            memory.clear()
            st.session_state["agent"] = ChatbotAgent(system_prompt=settings.system_prompt)
            st.rerun()
        st.subheader("Suggested prompts")
        st.markdown(
            """
Flow

1. Days — e.g. `4 days in Seoul`, `weekend in Seoul`, or just `5`.
2. Food — e.g. `BBQ + Gwangjang`, `chicken Hongdae`, `hanjeongsik Insadong`, or `anything`.
3. Places — e.g. `Gyeongbokgung, Bukchon, Hongdae`, `Gangnam + COEX`, or `anything`.
4. When the review screen appears in chat, reply once with confirm or ok to generate the itinerary.

Starters you can paste

- `Seoul trip, 3 days`
- `Plan Seoul for 5 days, I like street food and palaces`
- `2 days Seoul — food: tteokbokki + cafes, places: Myeongdong & Han River`
"""
        )

        with st.expander("Checkpoint Explorer"):
            if st.button("Load checkpoints", use_container_width=True):
                st.session_state["checkpoint_history"] = agent.get_state_history(limit=30)

            checkpoint_history = st.session_state.get("checkpoint_history", [])
            if checkpoint_history:
                options = [
                    f"#{row['index']} | next={row['next']} | {row['summary']}"
                    for row in checkpoint_history
                ]
                selected_option = st.selectbox("Select checkpoint", options=options)
                selected_index = options.index(selected_option)
                selected_checkpoint_id = checkpoint_history[selected_index]["checkpoint_id"]
                selected_checkpoint_ns = checkpoint_history[selected_index]["checkpoint_ns"]
                fork_prompt = st.text_input("Branch prompt", value="suggest a 2-day family trip in Seoul")
                if st.button("Run from checkpoint", use_container_width=True):
                    fork_response = agent.fork_from_checkpoint(
                        selected_checkpoint_id,
                        selected_checkpoint_ns,
                        fork_prompt,
                    )
                    memory.add("assistant", f"[FORK RESULT]\n{fork_response}")
                    st.rerun()

    # Enable PDF download only AFTER the itinerary message has been printed in chat
    # (i.e., stored in ChatMemory). This avoids enabling on intermediate state like
    # "Saved your food/place preference..." right after confirm/suggest steps.
    cached_itinerary = st.session_state.get("latest_itinerary_markdown")
    export_plan = cached_itinerary or _extract_latest_itinerary_from_memory(memory.all())

    with st.sidebar:
        st.subheader("Export")
        st.caption("Download a PDF from the latest itinerary.")
        pdf_name = "seoul-itinerary.pdf"
        pdf_bytes = b""
        pdf_error: str | None = None
        if export_plan:
            try:
                pdf_bytes = plan_markdown_to_pdf_bytes(
                    export_plan,
                    title=f"{settings.chatbot_title} — Itinerary",
                )
            except Exception as e:
                pdf_error = f"{type(e).__name__}: {e}"
                pdf_bytes = b""
        if pdf_error:
            st.error(f"PDF export failed: {pdf_error}")
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=pdf_name,
            mime="application/pdf",
            use_container_width=True,
            disabled=not bool(export_plan) or not bool(pdf_bytes),
            help=None if export_plan else "No itinerary in chat yet. Reply with confirm/ok to generate it.",
        )

    for msg in memory.all():
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    hints = agent.get_chat_input_hints()
    hint_panel_open = agent.waiting_for_human_review or any(
        "Suggest" in row["paste"]
        or row["paste"] in ("ok", "confirm")
        or "day" in row["paste"].lower()
        for row in hints
    )
    with st.expander(
        "💡 Quick paste (suggest food / places · confirm)",
        expanded=hint_panel_open,
    ):
        st.caption(
            "Copy a line into the chat box below. "
            "When a suggestion list appears: suggest food → choose a dish/restaurant; suggest places → choose a sightseeing spot/area."
        )
        for row in hints:
            st.code(row["paste"], language=None)
            st.caption(row["description"])

    user_input = st.chat_input("Describe your Seoul trip or answer the assistant...")
    if not user_input:
        return

    memory.add("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.status("Agent is processing your request...", expanded=True) as status:
            started_at = perf_counter()
            reply = agent.reply(user_input=user_input, history=memory.all())
            elapsed = perf_counter() - started_at
            status.update(
                label=f"Agent finished ({elapsed:.1f}s)",
                state="complete",
                expanded=False,
            )

        if elapsed >= 15:
            st.caption(
                "Generating the full itinerary can take a while. Quick follow-up messages are usually faster."
            )
        st.markdown(reply.text)
        if reply.next_actions:
            st.markdown("**Helpful next steps**")
            for item in reply.next_actions:
                st.markdown(f"- {item}")

    memory.add("assistant", _assistant_message_for_memory(reply))
    memory.trim(settings.max_history)
    if _is_itinerary_text(reply.text):
        if st.session_state.get("latest_itinerary_markdown") != reply.text:
            st.session_state["latest_itinerary_markdown"] = reply.text
            st.rerun()


if __name__ == "__main__":
    main()
