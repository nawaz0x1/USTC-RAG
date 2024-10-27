import streamlit as st
from utils.assistant import UniversityAssistant
from dotenv import load_dotenv

@st.cache_resource
def get_assistant():
    load_dotenv()
    assistant = UniversityAssistant(index_name="ustc-rag-2048")
    return assistant

st.title("USTC Admission Office Bot")
assistant = get_assistant()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = assistant.get_response(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})