from chatbot import *
import streamlit as st
from PIL import Image

st.set_page_config(page_title="BeeTax", page_icon=":robot:")

st.markdown("""
    <style>
    .custom-chat-input textarea {
        border-radius: 20px;
        padding: 10px 15px;
        border: 1px solid #ccc;
        resize: none;
        font-size: 16px;
        background-color: #f9f9f9;
    }
    </style>
""", unsafe_allow_html=True)

if "conversation" not in st.session_state:
  st.session_state.conversation = []

if "conversation" not in st.session_state:
  st.session_state.conversation = []

# st.header("BeeTax")
image = Image.open('./logo.png')
st.image(image, width=150)

# message = st.container(border=True, height=600)
if prompt := st.chat_input("Masukan pertanyaanmu", key="prompt"):

  for msg in st.session_state.conversation:
      with st.chat_message(msg["role"]):
          st.markdown(msg["content"])

  st.session_state.conversation.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  with st.chat_message("assistant"):
    message_placeholder = st.empty()
    message_placeholder.markdown("Loading...")
    answer = run_chatbot(prompt)
    message_placeholder.markdown(answer)

    st.session_state.conversation.append({"role": "assistant", "content": answer})



