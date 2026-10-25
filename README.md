# MP2: Constructing Langgraph RAG Agent with Ollama

This repository demonstrates how to construct a **Retrieval-Augmented Generation (RAG)** agent using **LangGraph** and **Ollama**.
The agent performs document retrieval following the same pipeline used in **MP1**, relying on **Pyserini** for retrieval.

## Prerequisites

- **Python** ≥ 3.10
- **Conda** (recommended)

## Installation

1. Use the environment configured in MP1. 

   The following steps assume that **Pyserini** has already been installed; if not, please refer to MP1 for installation instructions.
   Additionally, move the `data/` directory used in MP1 to the root of this project.

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Install Ollama 

   Follow the official instructions at https://ollama.com/download/. 
   For example, on linux:

   ```
   curl -fsSL https://ollama.com/install.sh | sh
   ```

4. Pull the Llama3.2 model (or another model of your choice):

   ```
   ollama pull llama3.2:3b
   ```

## Configuration

1. Create a `.env` file in the project root directory.

2. Set the following environment variables in the `.env` file:

   - `PYSERINI_CNAME`: The dataset we want to use to build indexes and retrieve documents (e.g., MP1 docs)
   - `OLLAMA_MODEL`: llama3.2:3b as default. You can change to the model you pull.
   - `RETRIEVER_K`: set the amount of docs u want to retrieve per query
   - `PYSERINI_K1`: The hyperparameter of bm25
   - `PYSERINI_B`: The hyperparameter of bm25

## Usage

Ensure your dataset is placed under the `data/` directory, then run:

   ``` 
   python main.py 
   ```

The agent will prompt you to enter a question. If you encounter an error such as `connection refused`, it means the Ollama service is not running.
Start it manually by running:

   ``` 
   ollama serve
   ```

Then, rerun the command above in a new terminal window.
## Project Structure

```
RAG-agents-Langgraph-demo/
├── data/               # Local document corpus (e.g., MP1 docs)
├── main.py             # Entry point
├── agent.py            # LangGraph agent logic
├── utils/
│   ├── config.py       # Configuration management
│   ├── llm.py          # Ollama LLM integration
│   ├── retriever.py    # Document retrieval using Pyserini
│   └── state.py        # State management for LangGraph
└── requirements.txt

```

## Acknowledgement

Part of the code in this repository is adapted from [RAG-agents-Langgraph](https://github.com/Feed-dev/RAG-agents-Langgraph/tree/main), 
which is licensed under the [MIT License](https://opensource.org/licenses/MIT).

We sincerely thank the original authors for their open-source contribution.