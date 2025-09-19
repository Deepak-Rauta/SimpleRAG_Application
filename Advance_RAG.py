import os
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import google.generativeai as genai
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
load_dotenv()

# Fetch the API key from the environment variables 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# -----------------------------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -----------------------------
# STEP 2: Helper Functions
# -----------------------------
def load_document(file_path):
    """Load text from a TXT or PDF file."""
    text = ""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    else:
        raise ValueError("Only .txt and .pdf files are supported")
    return text


def split_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)


def create_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings for chunks."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return model, embeddings


def build_faiss_index(embeddings):
    """Store embeddings in FAISS."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def retrieve(query, emb_model, cross_model, index, chunks, top_k=10, rerank_k=3):
    """Retrieve chunks with FAISS and rerank with CrossEncoder."""
    q_emb = emb_model.encode([query])
    distances, indices = index.search(q_emb, top_k)
    candidates = [chunks[i] for i in indices[0]]

    # Re-rank using cross-encoder
    pairs = [(query, c) for c in candidates]
    scores = cross_model.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    top_chunks = [c for c, _ in ranked[:rerank_k]]
    return "\n".join(top_chunks)


def generate_answer(question, context, history):
    """Use Gemini LLM with chat memory."""
    model = genai.GenerativeModel("models/gemini-1.5-flash")

    history_text = "\n".join([f"{role}: {msg}" for role, msg in history])

    prompt = f"""
    You are a helpful assistant. Use only the context below to answer.
    If the answer is not in the context, say "I don’t know."

    Conversation History:
    {history_text}

    Context:
    {context}

    Question: {question}
    Answer:
    """
    response = model.generate_content(prompt)
    return response.text

# -----------------------------
# STEP 3: Streamlit UI
# -----------------------------
st.set_page_config(page_title="Advanced RAG Chatbot", page_icon="🧠")

st.title("🧠 Advanced RAG Chatbot")
st.write("Upload a text/PDF file and chat with memory + reranking!")

uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

if uploaded_file is not None:
    file_path = "uploaded_file." + uploaded_file.name.split(".")[-1]
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("📖 Processing document...")
    text = load_document(file_path)
    chunks = split_text(text)

    st.write("⚡ Creating embeddings & FAISS index...")
    emb_model, embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)
    cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    st.write("✅ Ready! Ask me anything from your document.")

    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_input("Your question:")

    if st.button("Ask"):
        if question:
            context = retrieve(question, emb_model, cross_model, index, chunks)
            answer = generate_answer(question, context, st.session_state.history)
            st.session_state.history.append(("You", question))
            st.session_state.history.append(("Bot", answer))

    # Display history
    for speaker, msg in st.session_state.history:
        st.write(f"**{speaker}:** {msg}")
