from rag import ContextRetriever
from utils.jinjaProcessor import process_template
from langfuse import Langfuse
from langfuse.decorators import observe
from langfuse.openai import OpenAI
from dotenv import load_dotenv
import os
import re
load_dotenv()
langfuse = Langfuse()

class BeeTax:
    def __init__(self):
        self.llama_keys = [os.environ[f"LLM_API_KEY_{i}"] for i in range(1, 8) if f"LLM_API_KEY_{i}" in os.environ]
        self.llama_current_idx = 0
        self.client = OpenAI(api_key=self.llama_keys[self.llama_current_idx], base_url=os.environ["LLM_BASE_URL"])
        self.retriever = ContextRetriever(dbpath="vectorDB", dbname="beetax", filename="data/Buku_PPh2126_Release_20240108_reformatted.pdf")

    def _switch_llama_key(self):
        self.llama_current_idx = (self.llama_current_idx + 1) % len(self.llama_keys)
        return OpenAI(api_key=self.llama_keys[self.llama_current_idx], base_url=os.environ["LLM_BASE_URL"])

    @observe(name= "bot_response")
    def _infer(self,messages):
        try:
            response = self.client.chat.completions.create(
                model= os.environ["MODEL"],
                messages=messages,
                temperature=0.1,
                top_p=0.1
            )
            raw_output = response.choices[0].message.content.strip()

            # temporary solution to clean the output from <|...|> tags
            if "</think>" in raw_output:
                cleaned_output = raw_output.split("</think>", 1)[1].strip()
            else:
                cleaned_output = re.sub(r"<\|.*?\|>", "", raw_output).strip()

            return cleaned_output

        except Exception as e:
            self.client = self._switch_llama_key()

    def run(self):
        flag = False
        messages = []
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the chatbot. Goodbye!")
                break

            messages.append({"role": "user", "content": user_input})
            
            context = self.retriever.retrieve_contexts(messages, top_k=3)

            # Pass it to system prompt
            system_prompt = process_template("prompt/chatbot_system_prompt.jinja", {"context": context})
            if flag == False:
                messages.insert(0, {"role": "system", "content": system_prompt})
                flag = True
            else:
                messages[0] = {"role": "system", "content": system_prompt}

            # Get response from the LLM
            bot_response = self._infer(messages)

            print(f"Bot: {bot_response}",flush=True)

            messages.append({"role": "assistant", "content": bot_response})
        