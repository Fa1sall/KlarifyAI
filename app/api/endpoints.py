import time
import asyncio
from fastapi import APIRouter, HTTPException
from app.models.schema import QARequest, QAResponse
from app.services.pdf_utils import extract_text_from_pdf_url, chunk_text
from app.services.faiss_service import FaissService
from app.services.gemini_service import generate_answer
from db.crud import get_or_create_document, save_chunks, log_qa
from db.database import database

router = APIRouter()
faiss_cache: dict[str, FaissService] = {}

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

        # Build or load persistent FAISS index
        svc = faiss_cache.get(url)
        if svc is None:
            svc = FaissService()
            # Try loading existing
            if not svc.load(doc_id):
                svc.reset()
                svc.add_texts(chunks)
                svc.save(doc_id)
            faiss_cache[url] = svc
        else:
            # ensure in-memory index is loaded
            svc.load(doc_id)

        answers = []
        for question in req.questions:
            ctx_chunks = svc.get_relevant_context(question)
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
