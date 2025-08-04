import time
import asyncio
from fastapi import APIRouter, HTTPException
from app.models.schema import QARequest, QAResponse
from app.services.pdf_utils import extract_text_from_pdf_url, chunk_text
from app.services.pinecone_store import upsert_chunks, search_chunks
from app.services.gemini_service import generate_answer
from db.crud import get_or_create_document, save_chunks, log_qa
from db.database import database

router = APIRouter()

@router.post("/run", response_model=QAResponse)
async def run_klarify(req: QARequest):
    start = time.time()
    await database.connect()
    try:
        url = str(req.documents)
        raw_text = extract_text_from_pdf_url(url)
        doc_id = await get_or_create_document(url, raw_text)

        # First-time chunk save
        chunks = chunk_text(raw_text)
        await save_chunks(doc_id, chunks)

        # Add to Pinecone
        upsert_chunks(str(doc_id), chunks)

        # Answer questions
        answers = []
        for question in req.questions:
            ctx_chunks = search_chunks(question, top_k=5)
            context = "\n\n".join(ctx_chunks).strip()
            if not context:
                ans = "Not specified in the provided context."
            else:
                ans = await asyncio.to_thread(generate_answer, context, question)
            answers.append(ans)
            await log_qa(doc_id, question, ans)

        return QAResponse(
            answers=answers,
            processing_time=round(time.time() - start, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await database.disconnect()
