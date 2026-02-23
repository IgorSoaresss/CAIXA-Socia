from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.database import get_vectorstore

def ingest_data(urls=None, file_paths=None):
    docs = []

    # 1. Carregar Documentos
    if urls:
        print(f"🔄 Carregando URLs: {urls}...")
        loader = WebBaseLoader(urls)
        docs.extend(loader.load())

    if file_paths:
        for path in file_paths:
            print(f"🔄 Carregando Arquivo: {path}...")
            loader = TextLoader(path)
            docs.extend(loader.load())

    if not docs:
        print("❌ Nenhum documento para processar.")
        return
    
    # 2. Dividir Texto (Chunking)
    print("✂️ Dividindo textos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    # 3. Indexar no PGVector
    print(f"💾 Salvando {len(splits)} fragmentos no PGVector...")
    vectorstore = get_vectorstore()
    vectorstore.add_documents(splits)
    print("✅ Ingestão concluída!")

if __name__ == "__main__":
    # Exemplo de uso: Ingestão da documentação do LangChain como teste
    ingest_data(urls=["https://python.langchain.com/docs/introduction/"])
    