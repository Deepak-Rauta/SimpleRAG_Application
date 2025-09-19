import os
import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# Fetch the API key from the environment variables 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Configure the Generative AI client with the API key
genai.configure(api_key=GOOGLE_API_KEY)

def load_document(file_path):
    """Load and read the TXT orPDF document."""
    text = "" # Initialize the empty text variable
    if file_path.endswith("txt"): # Checks whether the filename ends with .txt. If true, the function treats the file as a plain text file.
        with open(file_path, "r", encoding="utf-8") as f: # Open the file in read mode with UTF-8 encoding
            text = f.read() # Reads the entire file contents as a single string and stores it in text.
    elif file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages: # Iterates over each page object in the PDF.
            text += page.extract_text() + "\n" # Extracts text from the current page and appends it to text, adding a newline character for separation.
    else:
        return ValueError("Unsupported file format. Please upload a TXT or PDF file.")
    return text

def split_text(text, chunk_size=500, overlap=100):
    """Split text into small overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = overlap
    )
    return splitter.split_text(text)

# Converts your text chunks into vector embeddings (numerical representations).
def create_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """Generating embeddings for chunks using SentenceTransformers."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return model, embeddings

def build_faiss_index(embeddings):
    """Store embeddings in a FAISS index."""
    dim = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve(query, model, index, chunks, top_k=3):
    """Retrive top relevant chunks from FAISS index."""
    q_emb = model.encode([query])
    distances, indices = index.search(q_emb, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return "\n".join(retrieved_chunks)

def generate_answer(question, context):
    """Use Gemini LLM to generate an answer from retrieved context."""
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    prompt = f"""
    You are a helpful assistant. Use only the context below to answer.

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
st.set_page_config(page_title="Simple RAG Chatbot", page_icon="🤖")

st.title("🤖 Simple RAG Chatbot")
st.write("Upload a text/PDF file and ask questions about it!")

# File upload
uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

if uploaded_file is not None:
    # Save uploaded file locally
    file_path = "uploaded_file." + uploaded_file.name.split(".")[-1]
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and split
    st.write("📖 Reading & splitting document...")
    text = load_document(file_path)
    chunks = split_text(text)

    # Embeddings + FAISS
    st.write("⚡ Creating embeddings & FAISS index...")
    emb_model, embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)

    # Chat interface
    st.write("✅ Ready! Ask me anything from your document.")

    # Store chat history in Streamlit session
    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_input("Your question:")

    if st.button("Ask"):
        if question:
            context = retrieve(question, emb_model, index, chunks)
            answer = generate_answer(question, context)
            st.session_state.history.append(("You", question))
            st.session_state.history.append(("Bot", answer))

    # Display chat history
    for speaker, msg in st.session_state.history:
        st.write(f"**{speaker}:** {msg}")