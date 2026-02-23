from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from src.config import Config

def get_vectorstore():
    embeddings = OllamaEmbeddings(
        model=Config.EMBED_MODEL,
        base_url=Config.OLLAMA_URL
    )

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=Config.COLLECTION_NAME,
        connection=Config.DB_URI,
        use_jsonb=True,
    )

    return vector_store
