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

from colors import Colors

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
def generate_subqueries(retriever, query, state):
    """
    Generate subqueries for the supplied question.
    This is to help us break down longer queries
    """
    # Generate subqueries
    llm = OllamaLLM(model=config.OLLAMA_MODEL)
    prompt = f"""Generate 3 distinct search queries that could help answer the question. Each query should cover different aspects of the question.
Avoid redundancy and ensure that the queries are relevant to the original question.
Ensure these queries are concise, relevant, and do not include any additional context or explanations.
Please format these queries as questions relevant to the original question.
Return only the queries, one per line. Return an empty response only if nothing is relevant.

Question: {query}
"""
    subqueries = llm.invoke(prompt).split("\n")
    state["subqueries"] = [q.strip() for q in subqueries if q.strip()]

    logger.info("==== Generated subqueries ====")
    logger.info("\n".join(state["subqueries"]))
    logger.info("==============================")

    # Retreive documents for each subquery and accumulate context
    docs = []
    seen_docs = set()
    added_context = []
    for i, subquery in enumerate(state["subqueries"]):
        # subquery_data = {"subquery": subquery, "results": []}
        logger.debug(f"Retrieving for subquery: {subquery}")
        # Retrieve documents for subquery (ContextualCompressionRetriever or BaseRetriever)
        subquery_docs = retriever.invoke(subquery)
        if not subquery_docs and hasattr(retriever, "base_retriever"):
            logger.info(
                "Subquery %r: compression yielded 0 docs; falling back to base retriever.",
                subquery,
            )
            subquery_docs = retriever.base_retriever.invoke(subquery)
        else:
            logger.info(
                "Subquery %r: retrieved %d docs", subquery, len(subquery_docs)
            )
        unique_docs = []
        for doc in subquery_docs:
            doc_id = getattr(doc, "metadata", {}).get("docid")
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)

                doc.metadata["subquery"] = subquery
                docs.append(doc)
                unique_docs.append(doc)
            else:
                logger.debug(
                    "Skipping duplicate or missing-docid document: %s",
                    getattr(doc, "metadata", {}),
                )
                continue

        added_context.append(f"Subquery {i+1}: {subquery}\n---\nRetrieved docs:\n" +
                                "\n".join(getattr(d, "page_content", str(d)) for d in unique_docs) + "\n---")

    ctx: List[str] = list(state.get("context") or []) + added_context
    return docs, ctx

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

        if len(query) > 100:
            # generate subqueries for long queries
            docs, ctx = generate_subqueries(retriever, query, state)
        else:
            # Retrieve documents (ContextualCompressionRetriever or BaseRetriever)
            docs = retriever.invoke(query)
            if not docs and hasattr(retriever, "base_retriever"):
                logger.info("Compression yielded 0 docs; falling back to base retriever.")
                docs = retriever.base_retriever.invoke(query)

            ctx: List[str] = list(state.get("context") or [])
            ctx.extend([getattr(d, "page_content", str(d)) for d in docs])

        # Update state with retrieved documents and context
        # this is the RETRIEVED DOCS metadata
        state["retrieved_documents"] = [
            {
                "subquery": d.metadata.get("subquery"),
                "text": d.page_content,
                "docid": d.metadata.get("docid"),
                "score": d.metadata.get("score"),
                "title": d.metadata.get("title"),
            }
            for d in sorted(
                docs,
                key=lambda doc: doc.metadata.get("score") or 0,
                reverse=True,
            )
        ]

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

Use ONLY the context below to answer the question.
Extract exact text from the document that may help answer the question. Include partially relevant documents if they may help.
Do not hallucinate or make up answers. If the answer is not in the context, say you don't know.

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
        # context_strings = "\n".join(context_strings)
        context_strings = "\n\n".join(
            f"--- Evidence group {i} ---\n{item}"
            for i, item in enumerate(state.get("context") or [], start=1)
        )
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
