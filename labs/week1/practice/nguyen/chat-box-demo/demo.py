from dotenv import load_dotenv
import base64
import os
import streamlit as st
import sqlite3
import uuid
from typing import TypedDict, Optional, Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from docx import Document
from PyPDF2 import PdfReader

load_dotenv()
llm = init_chat_model(f"openai:{os.getenv('OPENAI_MODEL_NAME')}")

PDF_OCR_MAX_PAGES = int(os.getenv("PDF_OCR_MAX_PAGES", "10"))


def _extract_pdf_text_layer(file_bytes: bytes) -> str:
    from io import BytesIO

    reader = PdfReader(BytesIO(file_bytes))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _extract_pdf_text_via_vision(file_bytes: bytes) -> str:
    import fitz

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    total = doc.page_count
    n = min(total, PDF_OCR_MAX_PAGES)
    chunks = []
    for i in range(n):
        page = doc[i]
        pix = page.get_pixmap(dpi=120)
        b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
        msg = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Extract every readable line of text from this PDF page image. "
                        "Preserve paragraph breaks. Output only the text, no preamble."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ]
        )
        r = llm.invoke([msg])
        chunk = r.content if hasattr(r, "content") else str(r)
        chunks.append((chunk or "").strip())
    doc.close()
    text = "\n\n".join(c for c in chunks if c)
    if total > n:
        text += (
            f"\n\n[... skipped {total - n} page(s); set PDF_OCR_MAX_PAGES to raise limit]"
        )
    return text

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

@st.cache_data(show_spinner="📂 Reading file...")
def parse_uploaded_file(file_bytes: bytes, file_type: str):
    file_type = file_type.lower()
    if file_type == "txt":
        return file_bytes.decode("utf-8", errors="ignore")
    elif file_type == "docx":
        from io import BytesIO
        doc = Document(BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs])
    elif file_type == "pdf":
        text = _extract_pdf_text_layer(file_bytes)
        if not text.strip():
            text = _extract_pdf_text_via_vision(file_bytes)
        return text
    return None

def detect_intent_and_language(state: dict):
    structured_llm = llm.with_structured_output(DetectLangOutput)

    prompt = (
        "You are a translation assistant.\n"
        "Extract:\n"
        "- src_lang: en | vi | ko (source language of the text to translate)\n"
        "- tgt_lang: en | vi | ko (target language the user wants)\n"
        "- content: ONLY the actual text to translate (no instruction words)\n\n"
        "Rules:\n"
        "- If the user says translate to ko / Korean / 한국어 / tiếng Hàn → tgt_lang must be ko.\n"
        "- If the user says translate to vi / Vietnamese / tiếng Việt → tgt_lang must be vi.\n"
        "- If the user says translate to en / English / tiếng Anh → tgt_lang must be en.\n"
        "- If the message has a 'User instruction:' line and 'Text to translate:' block, "
        "use the instruction for direction; put the block text in content.\n"
        "- If tgt_lang is clear from the user but src_lang is not, infer src_lang from the content.\n\n"
        f"Message:\n{state['user_input']}"
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

if "uploaded_file_content" not in st.session_state:
    st.session_state.uploaded_file_content = None

if "upload_key" not in st.session_state:
    st.session_state.upload_key = "file_uploader"

# --- SIDEBAR ---
if st.sidebar.button("🆕 New Chat"):
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.pending_translation = None
    st.session_state.uploaded_file_content = None
    st.session_state.upload_key = f"file_uploader_{str(uuid.uuid4())}"

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

with st.sidebar:
    uploaded_file = st.file_uploader(
        "📎 Upload .txt / .pdf / .docx",
        type=["txt", "docx", "pdf"],
        key=st.session_state.upload_key,
    )

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.read()
        file_content = parse_uploaded_file(file_bytes, uploaded_file.name.split(".")[-1])
        if not file_content or not file_content.strip():
            st.error("❌ File empty or unreadable.")
        else:
            st.session_state.uploaded_file_content = file_content
            with st.chat_message("assistant"):
                st.markdown(
                    f"📎 Uploaded: **{uploaded_file.name}**\n\n"
                    "👉 Now type: `translate to en` (or vi/ko)"
                )
    except Exception as e:
        st.error(f"❌ File error: {e}")

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
                    response = f"✅ **{translated}**" if translated else "❌ Failed"
                except Exception as e:
                    response = f"❌ Error: {e}"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        st.session_state.pending_translation = None
        st.session_state.uploaded_file_content = None

    elif is_translate_cmd:
        # if st.session_state.uploaded_file_content:
        #     input_text = (
        #         f"User instruction: {user_input}\n\n"
        #         f"Text to translate:\n{st.session_state.uploaded_file_content}"
        #     )
        # else:
        #     input_text = user_input
        words = user_input.strip().split()
        use_file = (
            st.session_state.uploaded_file_content
            and len(words) <= 3
        )
        
        if use_file:
            input_text = (
                f"User instruction: {user_input}\n\n"
                f"Text to translate:\n{st.session_state.uploaded_file_content}"
                )
            st.session_state.uploaded_file_content = None
        else:
            input_text = user_input

        with st.chat_message("assistant"):
            with st.spinner("🔍 Detecting..."):
                try:
                    detect_state = {"user_input": input_text}
                    detected = detect_intent_and_language(detect_state)
                    src = detected.get("src_lang")
                    tgt = detected.get("tgt_lang")
                    content = (detected.get("content") or "").strip()
                    if (
                        st.session_state.uploaded_file_content
                        and not content
                    ):
                        content = st.session_state.uploaded_file_content.strip()
                    st.session_state.pending_translation = {
                        "user_input": input_text,
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
        st.session_state.uploaded_file_content = None
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