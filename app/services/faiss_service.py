import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

# Directory to persist per-document FAISS indexes
INDEX_DIR = os.path.join(os.path.dirname(__file__), "../../indexes")
os.makedirs(INDEX_DIR, exist_ok=True)

class FaissService:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.texts: List[str] = []

    def reset(self):
        """Clear in-memory index and texts."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.texts = []

    def add_texts(self, chunks: List[str]):
        """Embed and add chunks to the FAISS index."""
        embs = self.embedder.encode(
            chunks, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        self.index.add(embs)
        self.texts.extend(chunks)

    def save(self, doc_id: int):
        """Persist index file and chunk texts for a given document ID."""
        idx_path = os.path.join(INDEX_DIR, f"{doc_id}.index")
        txt_path = os.path.join(INDEX_DIR, f"{doc_id}.chunks")
        faiss.write_index(self.index, idx_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            for chunk in self.texts:
                f.write(chunk.replace("\n", " ") + "\n<<<END>>>\n")

    def load(self, doc_id: int) -> bool:
        """Load persisted index and texts if they exist. Returns True if loaded."""
        idx_path = os.path.join(INDEX_DIR, f"{doc_id}.index")
        txt_path = os.path.join(INDEX_DIR, f"{doc_id}.chunks")
        if not (os.path.isfile(idx_path) and os.path.isfile(txt_path)):
            return False
        # Load index
        self.index = faiss.read_index(idx_path)
        # Load texts
        with open(txt_path, encoding="utf-8") as f:
            raw = f.read().split("\n<<<END>>>\n")
            self.texts = [r.strip() for r in raw if r.strip()]
        return True

    def get_relevant_context(self, query: str, top_k: int = 10) -> List[str]:
        """Retrieve top_k chunks most similar to query."""
        q_emb = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        D, I = self.index.search(q_emb, top_k)
        return [self.texts[i] for i in I[0] if 0 <= i < len(self.texts)]
