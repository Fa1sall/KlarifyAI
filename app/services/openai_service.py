import os
import openai
import asyncio
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = openai.OpenAI(api_key=openai.api_key)

async def get_answers_from_openai(context: str, question: str, timeout: float = 10.0) -> str:
    prompt = f"""Answer the question based only on the context below. Be concise and precise.

Context:
{context}

Question: {question}

Instructions:
- Provide a direct answer
- Use bold for key information
- Keep response under 100 words
- Only use information from context
"""
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.1,
            n=1,
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        logger.error(f"OpenAI error: {str(e)}")
        return f"**OpenAI Error**: {str(e)}"