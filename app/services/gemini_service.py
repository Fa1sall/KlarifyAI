import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set")

genai.configure(api_key=api_key)
model = genai.GenerativeModel(
    "gemini-2.0-flash",
    generation_config={
        "max_output_tokens": 200,
        "temperature": 0.0,
        "top_p": 0.8,
        "top_k": 40
    }
)

def generate_answer(context: str, question: str) -> str:
    prompt = f"""SYSTEM:
You are an expert insurance policy assistant.
Use ONLY the text in CONTEXT to answer.
If the answer is not in the CONTEXT, reply exactly:
Not specified in the provided context.
Do not use markdown, bullet points, or special formatting.

USER:
CONTEXT:
\"\"\"
{context}
\"\"\"

QUESTION:
{question}

RESPONSE:
"""
    resp = model.generate_content(prompt)
    ans = resp.text.strip()
    if not ans or ans.lower().startswith("not specified"):
        return "Not specified in the provided context."
    return ans
