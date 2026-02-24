from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from src.config import Config
import psycopg

def get_vectorstore():
    embeddings = OllamaEmbeddings(
        model=Config.EMBED_MODEL,
        base_url=Config.OLLAMA_URL
    )

    with psycopg.connect(Config.DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=Config.COLLECTION_NAME,
        connection=Config.DB_URI,
        use_jsonb=True,
    )

    return vector_store
