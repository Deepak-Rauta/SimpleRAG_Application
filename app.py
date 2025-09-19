import streamlit as st
import pypdf
from models.llm import get_gemini_response
from utils.rag import retrieve, build_vector_store, chunk_text

st.set_page_config(page_title="NeoStats RAG Chatbot", layout="wide")
st.title("📂 Document-based RAG Chatbot")

# --- Upload Section ---
uploaded_file = st.file_uploader("Upload a PDF or Text file", type=["pdf", "txt"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    else:
        text = uploaded_file.read().decode("utf-8")

    # Chunk and build vector DB
    chunks = chunk_text(text)
    build_vector_store(chunks, save=True)
    st.success("✅ Document processed and indexed!")

# --- Ask questions ---
mode = st.radio("Response Mode:", ["Concise", "Detailed"])
query = st.text_input("Ask a question based on uploaded document:")

if st.button("Ask") and query:
    context_docs = retrieve(query)
    context = "\n".join(context_docs)

    if mode == "Concise":
        prompt = f"Answer briefly using context:\n{context}\n\nQuestion: {query}\n"
    else:
        prompt = f"Answer in detail using context:\n{context}\n\nQuestion: {query}\n"

    answer = get_gemini_response(prompt)
    st.write("### Response:")
    st.write(answer)

    with st.expander("📑 Retrieved Context"):
        for doc in context_docs:
            st.markdown(f"- {doc}")