from pydantic import BaseModel, HttpUrl
from typing import List

class QARequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]
    processing_time: float
