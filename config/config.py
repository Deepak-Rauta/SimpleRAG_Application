import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Vector DB settings (FAISS for local)
VECTOR_DB_PATH = "vector_store"
