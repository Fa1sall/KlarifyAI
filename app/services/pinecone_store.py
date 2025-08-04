import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from typing import List

load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
model = SentenceTransformer("all-mpnet-base-v2")

def upsert_chunks(document_id: str, chunks: List[str]):
    vectors = []
    for i, chunk in enumerate(chunks):
        emb = model.encode(chunk, convert_to_numpy=True, normalize_embeddings=True)
        vectors.append({
            "id": f"{document_id}-{i}",
            "values": emb.tolist(),
            "metadata": {"text": chunk}
        })
    index.upsert(vectors=vectors)

def search_chunks(query: str, top_k: int = 5) -> List[str]:
    query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    result = index.query(vector=query_emb.tolist(), top_k=top_k, include_metadata=True)
    return [match['metadata']['text'] for match in result['matches'] if 'text' in match['metadata']]
