import re
from io import BytesIO
from typing import List
from PyPDF2 import PdfReader
import requests

def extract_text_from_pdf_url(pdf_url: str) -> str:
    """
    Download a PDF from the given URL and extract its full text,
    collapsing all whitespace to single spaces.
    """
    resp = requests.get(pdf_url, timeout=20)
    resp.raise_for_status()
    reader = PdfReader(BytesIO(resp.content))
    full_text = ""
    for page in reader.pages:
        t = page.extract_text() or ""
        full_text += re.sub(r"\s+", " ", t) + " "
    return full_text.strip()

def fallback_sent_tokenize(text: str) -> List[str]:
    """
    Simple sentence splitter: splits on punctuation followed by whitespace.
    """
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(text: str, chunk_size: int = 1600, overlap: int = 200) -> List[str]:
    """
    Split the text into overlapping chunks of roughly chunk_size characters,
    with an overlap of overlap characters between chunks.
    """
    sentences = fallback_sent_tokenize(text)
    chunks: List[str] = []
    curr_chunk: List[str] = []
    curr_len = 0

    for sent in sentences:
        sent_len = len(sent)
        # If a single sentence is longer than chunk_size, flush and split it
        if sent_len > chunk_size:
            if curr_chunk:
                chunks.append(" ".join(curr_chunk))
                curr_chunk = []
                curr_len = 0
            for i in range(0, sent_len, chunk_size):
                chunks.append(sent[i : i + chunk_size])
            continue

        # If adding this sentence would exceed chunk_size, flush the current chunk
        if curr_len + sent_len > chunk_size:
            if curr_chunk:
                chunks.append(" ".join(curr_chunk))
            # carry over up to 'overlap' characters worth of sentences
            overlap_sents: List[str] = []
            total = 0
            for s in reversed(curr_chunk):
                total += len(s)
                overlap_sents.insert(0, s)
                if total >= overlap:
                    break
            curr_chunk = overlap_sents
            curr_len = sum(len(s) for s in curr_chunk)

        # Add sentence to current chunk
        curr_chunk.append(sent)
        curr_len += sent_len

    # Flush any remaining sentences
    if curr_chunk:
        chunks.append(" ".join(curr_chunk))

    return chunks
