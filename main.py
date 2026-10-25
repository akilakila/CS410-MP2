import logging
from agent import create_agent
from utils.retriever import create_retriever
from utils.config import Config
from utils.state import GraphState

from colors import Colors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_output(result: GraphState):
    """
    Format and display the output from the agent.
    This function can be customized to improve user-facing output.
    """

    print(f"{Colors.HEADER}\n================ Retrieved Documents ================{Colors.RESET}")
    for idx, doc in enumerate(result.get("retrieved_documents", []), start=1):
        print(
            f"{Colors.HEADER}[{idx}]{Colors.RESET} {Colors.DOC_ID}docid={doc.get('docid')}{Colors.RESET} "
            f"{Colors.SCORE}BM25 score={doc.get('score')}{Colors.RESET} "
            f"{Colors.TITLE}title={doc.get('title') or 'Unknown'}{Colors.RESET}"
        )
        # This helps the user understand which subquery retrieved the document
        print(f"  {Colors.SUBQUERY}Retrieved for SUBQUERY: {doc.get('subquery', 'Unknown')}{Colors.RESET}")
        print(doc.get("text", ""))

def main():
    try:
        # Load configuration
        config = Config()
        logger.info(f"Loaded configuration: {config}")

        # Initialize components
        retriever = create_retriever()

        # Create agent
        agent = create_agent()

        while True:
            # Prompt user for question
            question = input("Please enter your question (or type 'exit' to quit): ")

            if question.lower() == 'exit':
                print("Exiting the program. Goodbye!")
                break

            logger.info(f"Running agent with question: {question}")

            # Initialize the GraphState with all required fields
            initial_state = GraphState(
                question=question,
                context=[],
                current_step="",
                final_answer="",
                retriever=retriever,
                web_search_tool=None,
                error=None,
                selected_namespaces=[],
                web_search_results=[],
                retrieved_documents=[],
                subqueries=[]
            )

            result = agent.invoke(initial_state)
            # lowering log level to debug to avoid cluttering the output with too much info
            logger.debug(f"Agent result: {result}")

            # ===== Task 2: this is where you can improve the user-facing output =====
            # You can show:
            # - retrieved passages (from result["context"])
            # - doc ids / scores (if you stored them in retrieval)
            # - then the final LLM answer
            # This makes it clear WHY the agent answered that way.

            format_output(result)

            # Print final answer
            if result.get("final_answer"):
                print(f"{Colors.HEADER}\nAnswer: {result['final_answer']}{Colors.RESET}")
            elif result.get("error"):
                print(f"{Colors.ERROR}\nError occurred: {result['error']}{Colors.RESET}")
            else:
                print(f"{Colors.HEADER}\nNo answer or error was returned.{Colors.RESET}")

            print("\n" + "-" * 50 + "\n")  # Add a separator between questions

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    main()
