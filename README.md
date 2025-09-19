# ![BeeTax_icon](res/logo.png)
Your personal chatbot that answers PPh (Pajak Penghasilan) related question

## Prerequisite
Installation using the provided requirements file.
```bash
pip install -r requirements.txt
```

Create `.env` file containing
```env
# EMBEDDING MODEL
HF_TOKEN = "<HUGGING_FACE_ACCESS_TOKEN>"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# LLM
MODEL = "deepseek-r1-distill-llama-70b"

# LANGFUSE
LANGFUSE_SECRET_KEY="sk-lf-8bb087fd-3d3c-4e35-9420-4cd27d8660b6"
LANGFUSE_PUBLIC_KEY="pk-lf-1276e49a-58ca-4c57-9d7c-46639688ead4"
LANGFUSE_HOST="http://localhost:3000"

# GEMINI
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/"
GEMINI_API_KEY_1 = "********************************"

# LLM
MODEL = "deepseek-r1-distill-llama-70b"
LLM_BASE_URL = "https://api.groq.com/openai/v1"

LLM_API_KEY_1 = "*********************************"
```

## Usage
```python
beetax = BeeTax()
beetax.run()
```
## Known Limitation
- Retrieved context are stacked together across different sub-queries. Each context needs to be labeled with the associated sub-queries used
- Calculation is a major flaw and needs further improvement by using function call

## Next Approach
- Create knowledge graph for table:
    - Status penduduk (e.g kawin/tidak kawin, jumlah tanggungan)
    - Lapisan Penghasilan Kena Pajak
    - TER A, B, C
- Modules and prompts for function calling
- Connect knowledge graph to function modueles    