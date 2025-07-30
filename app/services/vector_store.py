import numpy as np
import faiss
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from functools import lru_cache

@dataclass
class SearchResult:
    text: str
    score: float
    metadata: Optional[Dict] = None

class VectorStore:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        index_type: str = "flat",
        nlist: int = 4,
        nprobe: int = 2,
        cache_size: int = 1000
    ):
        if not hasattr(self, 'model'):
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            # REMOVE lru_cache here!
            self.encode = self.model.encode
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.texts = []
        self.metadata = []
        self._create_index()

    def _create_index(self):
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, 
                self.embedding_dim,
                self.nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            self.index.nprobe = self.nprobe
            self.needs_training = True

    def add_texts(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
        batch_size: int = 32
    ) -> bool:
        try:
            if not texts:
                return False
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                embeddings = self.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                all_embeddings.append(embeddings)
            embeddings = np.vstack(all_embeddings).astype(np.float32)
            if getattr(self, 'needs_training', False):
                if len(embeddings) >= self.nlist:
                    self.index.train(embeddings)
                    self.needs_training = False
                else:
                    return False
            self.index.add(embeddings)
            self.texts.extend(texts)
            self.metadata.extend(metadata or [{} for _ in texts])
            return True
        except Exception as e:
            print(f"Error adding texts to vector store: {str(e)}")
            return False

    def search(
        self,
        query: str,
        top_k: int = 6,
        threshold: float = 0.35
    ) -> List[SearchResult]:
        try:
            query_embedding = self.encode(
                [query],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32)
            actual_top_k = min(top_k * 2, len(self.texts))
            distances, indices = self.index.search(query_embedding, actual_top_k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.texts):
                    continue
                score = 1 / (1 + np.sqrt(max(0, dist)))
                if score < threshold:
                    continue
                results.append(SearchResult(
                    text=self.texts[idx],
                    score=score,
                    metadata=self.metadata[idx]
                ))
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []

    def get_relevant_context(
        self,
        query: str,
        max_length: int = 2000,
        min_score: float = 0.3
    ) -> Tuple[str, float]:
        results = self.search(query, top_k=5)
        if not results:
            return "", 0.0
        context_parts = []
        total_length = 0
        total_score = 0
        used_results = 0
        for result in results:
            if result.score < min_score:
                continue
            text_length = len(result.text)
            if total_length + text_length <= max_length:
                context_parts.append(result.text)
                total_length += text_length
                total_score += result.score
                used_results += 1
            else:
                break
        if not used_results:
            return "", 0.0
        avg_score = total_score / used_results
        return " ".join(context_parts), avg_score