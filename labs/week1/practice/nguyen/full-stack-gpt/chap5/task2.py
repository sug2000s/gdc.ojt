import os

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

st.set_page_config(
	page_title="Model Selector Chat",
	page_icon="🤖",
)

st.title("Model Selector Chat")

if "messages" not in st.session_state:
	st.session_state["messages"] = []

model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]

with st.sidebar:
	selected_model = st.selectbox("Choose a model", model_options, key="selected_model")
	if st.button("Clear Chat History"):
		st.session_state["messages"] = []


def get_llm(model_name):
	return ChatOpenAI(
		base_url=os.getenv("OPENAI_BASE_URL"),
		api_key=os.getenv("OPENAI_API_KEY"),
		model=model_name,
		temperature=0.1,
	)


def send_message(message, role, save=True):
	with st.chat_message(role):
		st.write(message)
	if save:
		st.session_state["messages"].append({"message": message, "role": role})


for message in st.session_state["messages"]:
	send_message(message["message"], message["role"], save=False)


user_message = st.chat_input("Send a message to the AI")

if user_message:
	send_message(user_message, "human")
	llm = get_llm(selected_model)
	response = llm.invoke(user_message)
	send_message(response.content, "ai")
