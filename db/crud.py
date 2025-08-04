from sqlalchemy import insert, select, and_
from db.database import database
from db.models import Document, Chunk, QARecord

async def get_or_create_document(url: str, raw_text: str):
    row = await database.fetch_one(select(Document).where(Document.url == url))
    if row:
        return row["id"]
    query = insert(Document).values(url=url, raw_text=raw_text)
    return await database.execute(query)

async def save_chunks(document_id: int, chunks: list[str]):
    values = [
        {"document_id": document_id, "chunk_text": chunk, "chunk_index": idx}
        for idx, chunk in enumerate(chunks)
    ]
    await database.execute_many(insert(Chunk), values)

async def log_qa(document_id: int, question: str, answer: str, score: float = None):
    query = select(QARecord).where(
        and_(
            QARecord.document_id == document_id,
            QARecord.question == question.strip(),
            QARecord.answer == answer.strip()
        )
    )
    exists = await database.fetch_one(query)
    if exists:
        return  
    insert_query = insert(QARecord).values(
        document_id=document_id,
        question=question.strip(),
        answer=answer.strip(),
        score=score
    )
    await database.execute(insert_query)
