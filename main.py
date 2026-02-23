from src.rag_graph import build_rag_graph

def main():
    app = build_rag_graph()

    print("💬 Sistema RAG Local Iniciado (digite 'sair' para encerrar)")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nVocê: ")
            if user_input.lower() in ["sair", "exit", "quit"]:
                break

            # Executa o grafo
            inputs = {"question": user_input}
            result = app.invoke(inputs)

            print(f"\n🤖 Assistente: {result['answer']}")

            # Opcional: Mostrar fontes
            # print("\n📚 Fontes:")
            # for doc in result['context']:
            #     print(f"- {doc.metadata.get('source', 'Desconhecido')}")

        except Exception as e:
            print(f"Erro: {e}")

if __name__ == "__main__":
    main()
