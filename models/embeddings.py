from sentence_transformers import SentenceTransformer

_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts):
    return _embedder.encode(texts)