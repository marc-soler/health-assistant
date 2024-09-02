from logging_setup import LoggerSetup
from elasticsearch_indexer import ElasticsearchIndexer
from prompt_builder import PromptBuilder
import prompt_templates as templates
from search_service import SearchService
from llm_service import LLMService
from response_evaluator import ResponseEvaluator
from cost_calculator import CostCalculator


class Orchestrator:
    """
    Orchestrator is responsible for coordinating the flow of data and interactions between various services,
    such as Elasticsearch, LLM, prompt generation, response evaluation, and cost calculation.
    """

    def __init__(self):
        self.logger = self.setup_logging()
        self.indexer = ElasticsearchIndexer()
        self.llm_service = LLMService()
        self.cost_calculator = CostCalculator()

    def setup_logging(self):
        """
        Sets up the logging configuration and returns a logger instance.
        """
        logger_setup = LoggerSetup()
        return logger_setup.get_logger()

    def run(self, context, question):
        """
        Runs the orchestration process, coordinating the interaction between the components.

        Args:
            context (str): The context for the question prompt.
            question (str): The question to be asked to the LLM.
        """
        try:
            # Initialize Elasticsearch and load data
            self.logger.info("Starting Elasticsearch indexing...")
            es_client, index_name = self.indexer.load_and_index_data()

            # Build the question prompt
            question_prompt_builder = PromptBuilder(templates.QUESTION_PROMPT_TEMPLATE)
            question_prompt = question_prompt_builder.build_prompt(
                context=context, question=question
            )
            self.logger.info(f"Question prompt built: {question_prompt}")

            # Search the Elasticsearch index
            search_service = SearchService(es_client=es_client)
            results = search_service.search(query=question_prompt)
            self.logger.info(f"Search results: {results}")

            # Connect to LLM and get a response
            self.llm_service.connect_to_llm()
            response, token_stats = self.llm_service.query_llm(prompt=question_prompt)
            self.logger.info(f"LLM response: {response}")

            # Evaluate the relevance of the response
            evaluation_prompt_builder = PromptBuilder(
                templates.EVALUATION_PROMPT_TEMPLATE
            )
            evaluator = ResponseEvaluator(
                llm_service=self.llm_service, prompt_builder=evaluation_prompt_builder
            )
            evaluation_result, token_stats = evaluator.evaluate_relevance(
                question, response
            )
            self.logger.info(f"Evaluation result: {evaluation_result}")

            # Calculate the cost of the LLM query
            cost = self.cost_calculator.calculate_openai_cost(token_stats)
            self.logger.info(f"OpenAI cost: ${cost:.6f}")

        except Exception as e:
            self.logger.error(
                f"An error occurred during the orchestration process: {e}"
            )


if __name__ == "__main__":
    orchestrator = Orchestrator()
    context = "The patient is a 45-year-old male with a history of hypertension."
    question = "What are the side effects of hypertension medication?"
    orchestrator.run(context=context, question=question)
