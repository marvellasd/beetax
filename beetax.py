from rag import ContextRetriever
from utils.jinjaProcessor import process_template
from langfuse import Langfuse
from langfuse.decorators import observe
from langfuse.openai import OpenAI
from dotenv import load_dotenv
from utils.jinjaProcessor import process_template_no_var
from utils.parser import extract_json_dict, parse_function, clean_response
from tools.calculate_tax import TaxCalculator
import json
import os
import re
load_dotenv()
langfuse = Langfuse()

class BeeTax(TaxCalculator):
    def __init__(self):
        super().__init__() 
        self.llama_keys = [os.environ[f"LLM_API_KEY_{i}"] for i in range(1, 8) if f"LLM_API_KEY_{i}" in os.environ]
        self.llama_current_idx = 0
        self.client = OpenAI(api_key=self.llama_keys[self.llama_current_idx], base_url=os.environ["LLM_BASE_URL"])
        self.retriever = ContextRetriever(dbpath="vectorDB", dbname="beetax", filename="data/Buku_PPh2126_Release_20240108_reformatted.pdf")

        self.function = {
            "calculate_tax_employee_should_pay": self.calculate_tax_employee_should_pay,
            "calculate_tax_company_should_pay": self.calculate_tax_company_should_pay
        }
        self.tools_prompt = process_template_no_var('prompt/tool_prompt.jinja')

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

    def _end_response(self,tool_call):
        end_response_func = ["NONE"]
        if tool_call['function_name'] in end_response_func:
            return True
        else:
            return False
      
    def generate_single_chat_message(self,user_prompt,messages,flag, user_intent):
        messages.append(
           { 
                "role": "user",
                "content": user_prompt
           }
        )

        # RAG is enabled and FC is disabled
        if user_intent == "knowledge_request":
            context = self.retriever.retrieve_contexts(messages)
            system_prompt = process_template('prompt/chatbot_system_prompt_rag.jinja', {"context": context})

            if flag == False:
                messages.insert(0, {"role": "system", "content": system_prompt})
                flag = True
            else:
                messages[0] = {"role": "system", "content": system_prompt}

            response = self._infer(messages)
            messages.append({
                "role": "assistant",
                "content": response
            })
            
            return messages, flag
        
        # RAG is disabled and FC is enabled
        else:
            temp = {
                "tool": self.tools_prompt
            }
            system_prompt = process_template('prompt/chatbot_system_prompt_tool_v2.jinja', temp) 

            if flag == False:
                messages.insert(0, {"role": "system", "content": system_prompt})
                flag = True
            else:
                messages[0] = {"role": "system", "content": system_prompt}

            while True:
                response = self._infer(messages)
                messages.append({
                    "role": "assistant",
                    "content": response
                })

                max_retries = 5 
                
                # Retry once if invalid format
                while max_retries > 0:
                    try:
                        llm_answer = clean_response(response,usage = "FC")
                        tools = parse_function(response)
                        break
                    except:
                        max_retries -=1
                        messages.append({
                            "role": "user",
                            "content": "invalid response format! **Follow** the format stated in your system prompt"
                        })
                        response = self._infer(messages)
                        messages.append({
                            "role": "assistant",
                            "content": response
                        })

                        # Remove unnecessary messages
                        del messages[-2]
                        del messages[-2]
                        

                if len(tools) == 1 and (tools[0]['function_name'] == 'FUNCTION_NOT_FOUND' or tools[0]['function_name'] == 'NONE'):
                    break
                            
                for tool in tools:

                    if tool['function_name'] == 'FUNCTION_NOT_FOUND' or tool['function_name'] == 'NONE' or tool['function_name'] == "determine_who_pays_the_tax":
                        return messages, flag
                
                    # Check for function name
                    try:
                        function_name = self.function[tool['function_name']]
                    except:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": "N/A",
                            "content": "Function not found in tools_dict"
                        })

                        continue

                    try:
                        function_args = tool['args']
                    except:
                        function_args = tool['parameters']

                    if "<MISSING>" in function_args.values() or function_args == {}:
                        return messages, flag

                    # Check for function arguments   
                    # try:
                    function_output = function_name(**function_args)
                    content = json.dumps(function_output, indent=4)
                    try:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": function_output['tool_call_id'],
                            "content": content
                        })
                    except Exception as e:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": "N/A",
                            "content": f"Error: {str(e)} calling {function_name.__name__} with args {function_args}"
                        })

            return messages, flag
        
    @observe(name = "Classify user intent")
    def _classify_user_intent(self, user_message, message):

        chat_history = [msg['content'] for msg in message[1:]]
        response = self._infer(
                [{
                    "role": "user",
                    "content": process_template("prompt/intent_classifier_prompt.jinja", {"chat_history": chat_history, "user_message": user_message})
                }]
            )
        response = extract_json_dict(response)
        return response['intent']

    def run_conversation(self, user_input, messages, flag):

        user_intent = self._classify_user_intent(user_input, messages)
            
        messages, flag = self.generate_single_chat_message(user_input, messages, flag, user_intent)
        return messages, flag

        
    