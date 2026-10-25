"""
agent.py
=========================
LangGraph-based RAG agent for MP2.

Pipeline:
1) retrieve(state)        -> collects context via Pyserini BM25 (+ optional compression)
2) generate_answer(state) -> prompts an LLM (Ollama) with the retrieved context

State keys (GraphState):
- question: str                     # user question (required)
- context: List[str]                # accumulated evidence passages
- retriever: Optional[BaseRetriever]# allow injection (for testing)
- final_answer: Optional[str]       # model's answer
- error: Optional[str]              # error message if any
- current_step: str                 # "retrieve" | "generate_answer"
"""

import logging
from typing import List

from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate

from utils.llm import OllamaLLM
from utils.retriever import create_retriever
from utils.state import GraphState
from utils.config import Config

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
config = Config()


# ---------------------------------------------------------------------
# Graph construction (entry for main.py)
# ---------------------------------------------------------------------
def create_agent():
    """
    Build and compile the LangGraph workflow:
       retrieve -> generate_answer -> END
    """
    workflow = StateGraph(GraphState)

    # Define nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)

    # Entry and edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


# ---------------------------------------------------------------------
# Node 1: Retrieval
# ---------------------------------------------------------------------
def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve evidence passages for the given question.

    Notes:
    - This is where retrieval happens for the basic single-query case.
    - If you later want to show retrieved documents, scores, or titles, keep the
      retrieved metadata here so the output layer can format it.
    - If you later add long / multi-sentence support, you can first break the
      question into several sub-queries here and retrieve for each of them, then
      merge the results into state["context"].
    """
    try:
        logger.info("Starting retrieval")

        query = (state.get("question") or "").strip()
        if not query:
            state["error"] = "Empty question"
            state["current_step"] = "retrieve"
            logger.warning("Retrieval aborted: empty question")
            return state

        # Allow DI for tests; otherwise create from config
        retriever = state.get("retriever") or create_retriever()

        # Generate subqueries
        llm = OllamaLLM(model=config.OLLAMA_MODEL)
        prompt = f"""Generate 3 distinct search queries that could help answer the question.
Return only the queries, one per line.

Question: {query}
"""
        subqueries = llm.invoke(prompt).split("\n")
        state["subqueries"] = [q.strip() for q in subqueries if q.strip()]

        # Retrieve documents (ContextualCompressionRetriever or BaseRetriever)
        docs = retriever.invoke(query)
        if not docs and hasattr(retriever, "base_retriever"):
            logger.info("Compression yielded 0 docs; falling back to base retriever.")
            docs = retriever.base_retriever.invoke(query)

        # Update state with retrieved documents and context
        state["retrieved_documents"] = [
            {
                "text": d.page_content,
                "docid": d.metadata.get("docid"),
                "score": d.metadata.get("score"),
                "title": d.metadata.get("title"),
            }
            for d in docs
        ]

        # Build/extend context
        ctx: List[str] = list(state.get("context") or [])
        ctx.extend([getattr(d, "page_content", str(d)) for d in docs])

        state["context"] = ctx
        state["current_step"] = "retrieve"
        logger.info(f"Retrieved {len(docs)} documents")
        return state

    except Exception as e:
        logger.exception("Error in retrieve")
        state["error"] = str(e)
        state["current_step"] = "retrieve"
        return state


# ---------------------------------------------------------------------
# Helpers for prompting
# ---------------------------------------------------------------------
def build_prompt(max_sentences: int = 7) -> ChatPromptTemplate:
    """
    Create the QA prompt.

    Notes:
    - You can experiment with different instruction styles here (e.g., chain-of-thought
      vs. concise answers, citing sources, etc.).
    """
    template = f"""You are an assistant for question answering.

Use ONLY the context below to answer the question. If the answer is not in the context, say you don't know.

Context:
{{context}}

Question:
{{question}}

Answer in at most {max_sentences} sentences, concise and to the point.
Answer:"""
    return ChatPromptTemplate.from_template(template)


# ---------------------------------------------------------------------
# Node 2: Answer generation
# ---------------------------------------------------------------------
def generate_answer(state: GraphState) -> GraphState:
    """
    Generate a concise answer using Ollama with the retrieved context.
    Safely stringifies all context items before prompting.

    Notes:
    - In Task 3, the retrieval step may store *structured* context in state["context"],
      e.g. a list like:
          [
            {"subquery": "what is RAG", "results": ["RAG is ...", "A RAG agent ..."]},
            {"subquery": "how do retrievers and vector DBs work together",
             "results": ["retriever uses embeddings ...", "vector DB stores ..."]},
            "Doc from baseline run"
          ]
      You can first turn each structured item into a text block like
          "Sub-query: ...\n---\nretrieved docs...\n---"
      in the retrieval step, which is compatible with the original format.
    """
    try:
        logger.info("Starting answer generation")

        # Initialize local LLM (Ollama)
        llm = OllamaLLM(model=config.OLLAMA_MODEL)

        ########## Prepare context string ##########
        # Flatten state["context"] into one string for the LLM.
        # If later you have multiple sub-queries, you can format them here.
        context_strings: List[str] = []
        for item in state.get("context") or []:
            context_strings.append(item if isinstance(item, str) else str(item))
        context_strings = "\n".join(context_strings)
        ###########################################

        # Build prompt and call the model
        prompt = build_prompt(max_sentences=7)
        chain = prompt | llm

        response = chain.invoke({
            "question": state.get("question", ""),
            "context": context_strings
        })

        # Update state
        state["final_answer"] = response
        state["current_step"] = "generate_answer"
        logger.info("Answer generation completed")
        return state

    except Exception as e:
        logger.error(f"Error in generate_answer: {e}")
        state["error"] = str(e)
        state["current_step"] = "generate_answer"
        return state
