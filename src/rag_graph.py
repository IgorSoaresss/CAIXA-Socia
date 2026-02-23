from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import START, END, StateGraph

from src.database import get_vectorstore
from src.config import Config

# 1. Definir o estado do Grafo
class RagState(TypedDict):
    question: str
    context: List[Document]
    answer: str

# 2. Definir os Nós
def retrieve(state: RagState):
    # Nó responsável apenas por buscar documentos relevantes
    print(f"🔎 Buscando contexto para: {state['question']}")
    vectorstore = get_vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(state["question"])
    
    return {"context": docs}

def generate(state: RagState):
    # Nó responsável por gerar a resposta usando o contexto
    print("🤖 Gerando resposta...")

    llm = ChatOllama(model=Config.LLM_MODEL, temperature=0)

    template = """Você é um assistente útil. Use os seguintes pedaços de contexto para responder à pergunta no final.
    Se você não souber a resposta, apenas diga que não sabe, não tente inventar uma resposta.
    Mantenha a resposta concisa.
    
    Contexto: {context}
    
    Pergunta: {question}
    
    Resposta:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Formata os docs em string única
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    chain = prompt | llm
    response = chain.invoke({"context": docs_content, "question": state["question"]})

    return {"answer": response.content}

# 3. Construir o grafo
def build_rag_graph():
    workflow = StateGraph(RagState)

    # Adiciona nós
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    # Define arestas (fluxo)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
