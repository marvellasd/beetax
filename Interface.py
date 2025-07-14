import streamlit as st
from PIL import Image
import pathlib
import re
import time
from chatbot import run_chatbot

def load_css(file_path):
   with open(file_path) as f:
      st.html(f"<style>{f.read()}</style>")

css_path = pathlib.Path("assets/style.css")
load_css(css_path)

st.set_page_config(page_title="BeeTax", page_icon=":robot:", layout="wide")

if "conversation" not in st.session_state:
  st.session_state.conversation = []

if "conversation" not in st.session_state:
  st.session_state.conversation = []

if 'show_container' not in st.session_state:
  st.session_state.show_container = True

if 'button_clicked' not in st.session_state:
  st.session_state.button_clicked = {"isClicked": False, "question": ""}

if 'load_answer' not in st.session_state:
  st.session_state.load_answer = {"reloaded": False, "question": ""}

image = Image.open('./logo.png')

if st.session_state.show_container:
  with st.container(key="ContainerB"):
    left_co, cent_co,last_co = st.columns([3, 1, 3])
    cent_co.image(image, use_container_width=True)
    
    st.markdown("<h3 style='text-align: center; margin-bottom: 10vh, margin-top: 2vh; backgroun-color: red;'>Halo, ada yang bisa saya bantu?</h3>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    left_space, left_button, left_space, cent_button, cent_space, last_button, last_space = st.columns([9, 3, 1, 3, 1, 3, 9])
    
    with left_button:
      if st.button("Literasi", use_container_width=True, key="left_Button"):
        st.session_state.button_clicked["isClicked"] = True
        st.session_state.button_clicked["question"] = "Apa itu pajak dan kenapa harus membayarnya?"

    with cent_button:
      if st.button("Hitung Pajak", use_container_width=True, key="cent_Button"):
        st.session_state.button_clicked["isClicked"] = True
        st.session_state.button_clicked["question"] = "Bagaimana cara menghitung pajak?"

    with last_button:
      if st.button("Cara Pembayaran", use_container_width=True, key="last_Button"):
        st.session_state.button_clicked["isClicked"] = True
        st.session_state.button_clicked["question"] = "Bagaimana metode pembayaran pajak?"
else:
  st.image(image, width=150)

for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def loadAnswer(prompt):
  if st.session_state.show_container == True:
      st.session_state.show_container = False
      st.session_state.load_answer["reloaded"] = True
      st.session_state.load_answer["question"] = prompt
      st.rerun()

  st.session_state.conversation.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  with st.chat_message("assistant"):
    message_placeholder = st.empty()
    message_placeholder.markdown("Loading...")
    answer = run_chatbot(prompt)

    split = re.findall(r'\S+|\s+', answer)
    typing = ""

    for word in split:
      typing += word
      message_placeholder.markdown(typing)
      time.sleep(0.002)

    st.session_state.conversation.append({"role": "assistant", "content": answer})

if st.session_state.button_clicked["isClicked"] == True:
  #  st.session_state.show_container = False
   loadAnswer(st.session_state.button_clicked["question"])
   st.session_state.button_clicked["isClicked"] = False
   st.session_state.button_clicked["question"] = ""

   if st.session_state.load_answer["reloaded"] == True:
     st.session_state.load_answer["reloaded"] = False
     st.session_state.load_answer["question"] = ""
elif st.session_state.load_answer["reloaded"] == True:
  loadAnswer(st.session_state.load_answer["question"])
  st.session_state.load_answer["reloaded"] = False
  st.session_state.load_answer["question"] = ""

if prompt:= st.chat_input("Masukan pertanyaanmu", key="prompt"):
  # st.session_state.show_container = False
  loadAnswer(prompt)