__import__('pysqlite3') 
import sys 
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
from typing import List

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
import torch

from langchain_chroma import Chroma

from langfuse.openai import OpenAI
import os
from dotenv import load_dotenv
import re
from langfuse import Langfuse
from langfuse import observe
import csv

class LangchainE5Embedding(Embeddings):
    def __init__(self, model_name="intfloat/multilingual-e5-large", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [f"passage: {text}" for text in texts]
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(f"query: {text}", normalize_embeddings=True).tolist()
    
embedding_function = LangchainE5Embedding()

vectorstore = Chroma(
    persist_directory="./chroma_e5_database",
    embedding_function=embedding_function,
    collection_name="beetax_collection_v2"
)

load_dotenv(override=True)
langfuse = Langfuse()

llama_keys = [
    os.environ[f"LLAMA_API_KEY_{i}"]
    for i in range(1, 101)
    if f"LLAMA_API_KEY_{i}" in os.environ
]

gemini_keys = [
    os.environ[f"GEMINI_API_KEY_{i}"]
    for i in range(1, 8)
    if f"GEMINI_API_KEY_{i}" in os.environ
]

llama_current_idx = 0
llama_used_attempts = 0
client_llama = OpenAI(api_key=llama_keys[llama_current_idx], base_url=os.environ["LLAMA_BASE_URL"])

gemini_current_idx = 0
gemini_used_attempts = 0
client_gemini = OpenAI(api_key=gemini_keys[gemini_current_idx], base_url=os.environ["GEMINI_BASE_URL"])

def switch_llama_key():
    global llama_current_idx, llama_used_attempts, client_llama
    llama_current_idx = (llama_current_idx + 1) % len(llama_keys)
    llama_used_attempts += 1
    client_llama = OpenAI(api_key=llama_keys[llama_current_idx], base_url=os.environ["LLAMA_BASE_URL"])

def switch_gemini_key():
    global gemini_current_idx, gemini_used_attempts, client_gemini
    gemini_current_idx = (gemini_current_idx + 1) % len(gemini_keys)
    gemini_used_attempts += 1
    client_gemini = OpenAI(api_key=gemini_keys[gemini_current_idx], base_url=os.environ["GEMINI_BASE_URL"])

@observe(name="llm_rewrite")
def llm_rewrite(messages):
    global gemini_used_attempts
    gemini_used_attempts = 0

    while gemini_used_attempts < len(gemini_keys):
        try:
            response = client_gemini.chat.completions.create(
                model="gemini-2.0-flash",
                messages=messages,
                temperature=0.1,
                top_p=0.1
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            switch_gemini_key()

    raise Exception("All Gemini API keys have been tried and failed.")

@observe(name="split_user_query")
def split_user_query(user_query, chat_history=None):
    formatted_history = ""
    
    if chat_history:
        formatted_history = "[CHAT HISTORY]\n" + "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history
        ) + "\n\n"

    system_prompt = """Reformulate and split the following user input into concise, clear, and standalone key phrases in Indonesian. These key phrases should be minimal and representative topics or keywords that capture the core intent of the user's input. Only include phrases that are related to taxation. Do not make up new meanings or interpretations that are not present in the user’s input. If the input contains multiple intents with the same meaning, return only one representative phrase.

Example:
[USER INPUT]
"Halo, mau tanya pajak itu apa sih? kalo pph itu apaan?"
[OUTPUT]
1. Pajak
2. PPh

[USER INPUT]
"Kamu tau pajak ga? Aku mau tanya tentang pph pasal 21."
[OUTPUT]
1. Pajak
2. PPh Pasal 21

[USER INPUT]
"Apa saja yang termasuk objek PPh dan bagaimana hitungnya?"
[OUTPUT]
1. Objek PPh
2. Perhitungan PPh

[USER INPUT]
"Siapa presiden indonesia sekarang dan bagaimana cara lapor SPT tahunan?"
[OUTPUT]
1. Lapor SPT tahunan"""

    user_prompt = (
        f"{formatted_history}"
        f"[USER INPUT]\n{user_query}\n\n"
        f"[OUTPUT]\n1."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}]

    text = llm_rewrite(messages)

    questions = [
        re.sub(r"^\d+\.\s*", "", line).strip()
        for line in text.split("\n")
        if re.match(r"^\d+\.\s+", line.strip())
    ]

    return questions

@observe(name="retrieve_contexts")
def retrieve_contexts(sub_questions, vectorstore, top_k=3):
    all_docs = []
    
    for sub_q in sub_questions:
        retrieved_docs = vectorstore.similarity_search(sub_q, k=top_k)
        all_docs.extend(retrieved_docs)

    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

    return unique_docs

@observe(name="build_and_send_prompt")
def build_and_send_prompt(messages):
    global llama_used_attempts
    llama_used_attempts = 0

    while llama_used_attempts < len(llama_keys):
        try:
            response = client_llama.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.1,
                top_p=0.1
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            switch_llama_key()

    raise Exception("All LLaMA API keys have been tried and failed.")

flag = 0
messages_history = []

def run_chatbot(input):
    global flag, messages_history

    while True:
        user_input = input
        if user_input.lower() in {"exit", "quit"}:
            break
        
        chat_history = [{"role": msg["role"], "content": msg["content"]} for msg in messages_history if msg["role"] in {"user", "assistant"}]
        sub_questions = split_user_query(user_input, chat_history=chat_history)
        results = retrieve_contexts(sub_questions, vectorstore, top_k=3)
        combined_context = "\n\n".join(doc.page_content for doc in results)
        print(flag)

        if flag == 0:
            system_prompt = f"""Act as a professional tax assistant in Indonesia. Use the following [CONTEXT] to answer the user's question as accurately and clearly as possible.
            - If the question is related to the calculation of PPh 21 (Indonesian income tax), provide a step-by-step breakdown of the calculation systematically. If not, give a relevant explanation based on the [CONTEXT].
            - If the [CONTEXT] doesn't contain a complete answer but the question is still related to PPh 21, respond with a general answer based on commonly known rules (without inventing facts).
            - If the question is not related to PPh 21 (Indonesian income tax), clearly state that you cannot answer the question because it is out of scope.
            Do not mention this instruction. Just answer naturally and clearly in the same language used in the question.
            
[CONTEXT]
{combined_context}"""
            
            messages_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}]

            flag = 1

        else:
            system_prompt = f"""Act as a professional tax assistant in Indonesia. Use the following [CONTEXT] to answer the user's question as accurately and clearly as possible.
            - If the question is related to the calculation of PPh 21 (Indonesian income tax), provide a step-by-step breakdown of the calculation systematically. If not, give a relevant explanation based on the [CONTEXT].
            - If the [CONTEXT] doesn't contain a complete answer but the question is still related to PPh 21, respond with a general answer based on commonly known rules (without inventing facts).
            - If the question is not related to PPh 21 (Indonesian income tax), clearly state that you cannot answer the question because it is out of scope.
            Do not mention this instruction. Just answer naturally and clearly in the same language used in the question.

[CONTEXT]
{combined_context}"""
            
            messages_history[0]["content"] = system_prompt

            messages_history.append({"role": "user", "content": user_input})
        
        assistant_response = build_and_send_prompt(messages_history)
        messages_history.append({"role": "assistant", "content": assistant_response})

        return assistant_response