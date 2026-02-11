import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def construir_agente():
    print("--- 1. Carregando a Base de Conhecimento ---")
    # Carrega o arquivo de texto que criamos
    loader = TextLoader("regras_bolsa_familia.txt", encoding="utf-8")
    documentos = loader.load()

    print("--- 2. Fragmentando o Texto (Chunking) ---")
    # LLMs têm limite de contexto. Dividimos o texto em pedaços menores.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    textos_divididos = text_splitter.split_documents(documentos)

    print("--- 3. Criando a Memória Vetorial (Embeddings) ---")
    # Transforma texto em números (vetores) para busca semântica
    embeddings = OpenAIEmbeddings()
    
    # Cria o banco de dados vetorial temporário (na memória)
    db = Chroma.from_documents(textos_divididos, embeddings)

    print("--- 4. Configurando o Cérebro (LLM) ---")
    # Usamos o GPT-3.5 ou GPT-4 com temperatura 0 para ser mais "exato" e menos criativo
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Cria a corrente de "Pergunta e Resposta com Recuperação" (RAG)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" apenas insere o texto no prompt
        retriever=db.as_retriever()
    )
    
    return qa_chain

# --- Execução do Agente ---
if __name__ == "__main__":
    agente = construir_agente()
    
    print("\n✅ Agente CAIXA pronto! Pergunte sobre o Bolsa Família (digite 'sair' para encerrar).")
    
    while True:
        pergunta = input("\nVocê: ")
        if pergunta.lower() in ["sair", "exit", "quit"]:
            break
            
        # O agente busca no texto e responde
        resposta = agente.invoke(pergunta)
        print(f"Agente: {resposta['result']}")
