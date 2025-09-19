import faiss
import numpy as np
import os
import pickle
from models.embeddings import get_embeddings
from config.config import VECTOR_DB_PATH

def build_vector_store(docs, save=True):
    vectors = get_embeddings(docs)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))

    # Save part of vector store
    if save:
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        faiss.write_index(index, os.path.join(VECTOR_DB_PATH, "faiss.index"))
        with open(os.path.join(VECTOR_DB_PATH, "docs.pkl"), "wb") as f:
            pickle.dump(docs, f)
    return index, docs

    # Load the vector store
def load_vector_store():
    index = faiss.read_index(os.path.join(VECTOR_DB_PATH, "faiss.index"))
    with open(os.path.join(VECTOR_DB_PATH, "docs.pkl"), "rb") as f:
        docs = pickle.load(f)
    return index, docs

def retrieve(query, k=3):
    index, docs = load_vector_store()
    q_vec = get_embeddings([query])
    D, I = index.search(np.array(q_vec), k)
    return [docs[i] for i in I[0]]

def chunk_text(text, chunk_size=300, overlap=50):
    """Split text into chunks with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


