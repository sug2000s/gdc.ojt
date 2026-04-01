from dotenv import load_dotenv
import os
import streamlit as st
import sqlite3
import uuid
from typing import TypedDict, Optional, Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.chat_models import init_chat_model

load_dotenv()
llm = init_chat_model(f"openai:{os.getenv('OPENAI_MODEL_NAME')}")

class TranslateState(TypedDict):
    user_input: str
    src_lang: Optional[Literal["en", "vi", "ko"]]
    tgt_lang: Optional[Literal["en", "vi", "ko"]]
    content: Optional[str]
    translated: Optional[str]
    tone: Optional[str]
    thread_id: str

class DetectLangOutput(BaseModel):
    src_lang: Literal["en", "vi", "ko"]
    tgt_lang: Literal["en", "vi", "ko"]
    content: str

class TranslateOutput(BaseModel):
    translated: str

def detect_intent_and_language(state: dict):
    structured_llm = llm.with_structured_output(DetectLangOutput)

    prompt = (
        "You are a translation assistant.\n"
        "Extract:\n"
        "- src_lang: en | vi | ko\n"
        "- tgt_lang: en | vi | ko\n"
        "- content: text only\n\n"
        f"User: {state['user_input']}"
    )

    try:
        res = structured_llm.invoke(prompt).model_dump()
    except Exception:
        res = {
            "src_lang": "en",
            "tgt_lang": "en",
            "content": state["user_input"]
        }

    return res

# def translate_text(state: dict):
#     src = state.get("src_lang")
#     tgt = state.get("tgt_lang")
#     text = state.get("content")

#     structured_llm = llm.with_structured_output(TranslateOutput)

#     prompt = f"Translate from {src} to {tgt}:\n{text}"

#     res = structured_llm.invoke(prompt)
#     state["translated"] = res.translated
#     return state

def translate_text(state: dict):
    src = state.get("src_lang")
    tgt = state.get("tgt_lang")
    text = state.get("content")
    tone = (state.get("tone") or "").strip()

    prompt = (
        f"You are a professional translator.\n"
        f"Translate the following text from {src} to {tgt}.\n"
    )
    if tone:
        prompt += (
            f"Apply this tone/register throughout (word choice, formality, style): {tone}\n"
        )
    prompt += f"Only return the translated text, no explanation.\n\nText:\n{text}"

    res = llm.invoke(prompt)
    print("[DEBUG: src]", src)
    print("[DEBUG: tgt]", tgt)
    print("[DEBUG: tone]", tone)
    print("[DEBUG: text]", text)

    # xử lý content
    translated = res.content if hasattr(res, "content") else str(res)

    state["translated"] = translated.strip()
    return state

graph_builder = StateGraph(TranslateState)
graph_builder.add_node("detect", detect_intent_and_language)
graph_builder.add_node("translate", translate_text)

graph_builder.add_edge(START, "detect")
graph_builder.add_edge("detect", "translate")
graph_builder.add_edge("translate", END)

conn = sqlite3.connect("memory.db", check_same_thread=False)
graph = graph_builder.compile(checkpointer=SqliteSaver(conn))

def chat_with_ai(messages, thread_id: str):
    res = llm.invoke(
        messages,
        config={"configurable": {"thread_id": thread_id}}
    )
    return res.content if hasattr(res, "content") else str(res)

# --- UI ---
st.set_page_config(page_title="🌐 AI Translator", layout="centered")
st.title("🌐 AI Translator")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "pending_translation" not in st.session_state:
    st.session_state.pending_translation = None

if "debug" not in st.session_state:
    st.session_state.debug = False

st.sidebar.checkbox("Show Debug", key="debug")

if st.sidebar.button("🆕 New Chat"):
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.pending_translation = None

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    is_translate_cmd = user_input.lower().startswith(("dịch", "translate", "번역"))

    if st.session_state.pending_translation is not None and not is_translate_cmd:
        pending = st.session_state.pending_translation
        tone = user_input.strip()
        state = {
            "user_input": pending.get("user_input", ""),
            "src_lang": pending.get("src_lang"),
            "tgt_lang": pending.get("tgt_lang"),
            "content": pending.get("content"),
            "translated": None,
            "tone": tone,
            "thread_id": st.session_state.thread_id,
        }
        with st.chat_message("assistant"):
            with st.spinner("🌐 Translating..."):
                try:
                    out = translate_text(dict(state))
                    translated = out.get("translated", "")
                    if not translated:
                        response = "❌ Could not translate."
                    else:
                        response = (
                            f"✅ **{translated}**"
                        )
                except Exception as e:
                    response = f"❌ Error: {e}"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        st.session_state.pending_translation = None

    elif is_translate_cmd:

        with st.chat_message("assistant"):
            with st.spinner("🔍 Detecting languages..."):
                try:
                    detect_state = {"user_input": user_input}
                    detected = detect_intent_and_language(detect_state)
                    src = detected.get("src_lang")
                    tgt = detected.get("tgt_lang")
                    content = detected.get("content") or ""
                    preview = content if len(content) <= 200 else content[:200] + "…"
                    st.session_state.pending_translation = {
                        "user_input": user_input,
                        "src_lang": src,
                        "tgt_lang": tgt,
                        "content": content,
                    }
                    response = (
                        "Bạn muốn dịch theo **sắc thái** nào? "
                        "(ví dụ: trang trọng, thân mật, marketing, văn phòng…)\n\n"
                        "What **tone** should we use? "
                        "(e.g. formal, casual, marketing, friendly…)\n\n"
                        "어떤 **톤**으로 번역할까요? "
                        "(예: 격식체, 반말, 마케팅, 업무 …)"
                    )
                except Exception as e:
                    st.session_state.pending_translation = None
                    response = f"❌ Error: {e}"

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    response = chat_with_ai(
                        st.session_state.messages,
                        st.session_state.thread_id
                    )
                except Exception as e:
                    response = f"❌ Error: {e}"

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})