from dotenv import load_dotenv
import os

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_classic.storage import LocalFileStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import BaseCallbackHandler
from openai import NotFoundError
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


MODEL_DEPLOYMENT_ENV = {
    "gpt-5.1": "OPENAI_DEPLOYMENT_GPT_5_1",
    "gpt-4o-mini": "OPENAI_DEPLOYMENT_GPT_4O_MINI",
    "gpt-4o": "OPENAI_DEPLOYMENT_GPT_4O",
    "gpt-4-turbo": "OPENAI_DEPLOYMENT_GPT_4_TURBO",
}


def resolve_model_name(selected_model):
    deployment_env = MODEL_DEPLOYMENT_ENV[selected_model]
    return os.getenv(deployment_env, selected_model)


def get_llm(model_name):
    return ChatOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=model_name,
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file_name, file_content):
    files_cache_dir = "./.cache/files"
    embeddings_cache_root = "./.cache/embeddings"
    os.makedirs(files_cache_dir, exist_ok=True)
    os.makedirs(embeddings_cache_root, exist_ok=True)
    file_path = os.path.join(files_cache_dir, file_name)
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(os.path.join(embeddings_cache_root, file_name))
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    file_ext = os.path.splitext(file_name)[1].lower()
    if file_ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        base_url=os.getenv("OPENAI_EMBEDDING_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_EMBEDDING_MODEL"),
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


st.title("DocumentGPT")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.markdown(
    """
Welcome!

Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    model_options = ["gpt-5.1", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
    default_model = os.getenv("OPENAI_MODEL_NAME", "gpt-5.1")
    default_model_index = (
        model_options.index(default_model) if default_model in model_options else 0
    )
    selected_model = st.selectbox(
        "Choose model",
        options=model_options,
        index=default_model_index,
    )
    selected_model_name = resolve_model_name(selected_model)
    st.caption(f"Model/Deployment: {selected_model_name}")

    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )
    if st.button("Clear Chat History"):
        st.session_state["messages"] = []

message = st.chat_input(
    "Ask anything about your file...",
    disabled=file is None,
)

if file:
    retriever = embed_file(file.name, file.getvalue())
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    if message:
        send_message(message, "human")
        llm = get_llm(selected_model_name)
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            try:
                chain.invoke(message)
            except NotFoundError:
                st.error(
                    "Deployment not found. For Azure OpenAI, set deployment env vars: "
                    "OPENAI_DEPLOYMENT_GPT_4O_MINI, OPENAI_DEPLOYMENT_GPT_4O, "
                    "OPENAI_DEPLOYMENT_GPT_4_TURBO."
                )


else:
    st.session_state["messages"] = []
    st.info("Please upload a file to start chatting.")
