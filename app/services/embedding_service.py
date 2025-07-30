from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed(self, texts):
        """
        Generate embeddings for given text(s)
        Args:
            texts: Single string or list of strings
        Returns:
            numpy array of embeddings
        """
        return self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )