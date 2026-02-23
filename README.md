# Como rodar o projeto

1.	Pré-requisitos:
○	Docker instalado.
○	Ollama instalado e rodando (ollama serve).
○	Modelos baixados:
Bash
ollama pull llama3
ollama pull nomic-embed-text

2.	Iniciar o Banco:
Bash
docker-compose up -d

3.	Instalar dependências Python:
Bash
pip install -r requirements.txt

4.	Ingerir Dados:
Rode o script de ingestão uma vez para popular o banco.
Bash
python -m src.ingest

5.	Rodar o Chat:
Bash
python main.py
