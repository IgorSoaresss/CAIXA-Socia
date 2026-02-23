import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DB_URI = os.getenv("POSTGRES_CONNECTION_STRING")
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
    COLLECTION_NAME = "fonte_informacoes"
    