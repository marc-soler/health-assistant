import uuid
from logging_setup import LoggerSetup
from database_manager import DatabaseManager
from elasticsearch_indexer import ElasticsearchIndexer
from prompt_builder import PromptBuilder
import prompt_templates as templates
from search_service import SearchService
from llm_service import LLMService
from response_evaluator import ResponseEvaluator
from cost_calculator import CostCalculator


class QueryOrchestrator:
    def __init__(self, es_client, llm_service):
        self.es_client = es_client
        self.llm_service = llm_service

    def process_query(self, context, question):
        search_service = SearchService(es_client=self.es_client)
        results = search_service.search(query=question_prompt)

        # TODO: fix context
        question_prompt_builder = PromptBuilder(templates.QUESTION_PROMPT_TEMPLATE)
        question_prompt = question_prompt_builder.build_prompt(
            context=context, question=question
        )

        response, token_stats = self.llm_service.query_llm(prompt=question_prompt)

        return response, token_stats


class EvaluationOrchestrator:
    def __init__(self, llm_service):
        self.llm_service = llm_service

    def evaluate_response(self, question, answer):
        evaluator = ResponseEvaluator(
            llm_service=self.llm_service,
            prompt_builder=PromptBuilder(templates.EVALUATION_PROMPT_TEMPLATE),
        )
        evaluation_result, token_stats = evaluator.evaluate_relevance(question, answer)
        return evaluation_result, token_stats


class FeedbackOrchestrator:
    def __init__(self, db_manager):
        self.db_manager = db_manager

    def submit_feedback(self, conversation_id, feedback, additional_feedback):
        # TODO: revise feedback
        feedback_value = 1 if feedback == "Like" else -1
        self.db_manager.save_feedback(
            conversation_id=conversation_id,
            feedback=feedback_value,
            timestamp=None,
        )
        if additional_feedback:
            # Save additional feedback as a comment or similar
            pass


class CostCalculationOrchestrator:
    def __init__(self):
        self.cost_calculator = CostCalculator()

    def calculate_cost(self, token_stats):
        cost = self.cost_calculator.calculate_openai_cost(token_stats)
        return cost


# Main Orchestrator that integrates all sub-orchestrators
class MainOrchestrator:
    def __init__(self):
        logger_setup = LoggerSetup()
        self.logger = logger_setup.get_logger()

        self.db_manager = DatabaseManager()

        indexer = ElasticsearchIndexer()
        self.es_client, self.index_name = indexer.load_and_index_data()

        self.llm_service = LLMService()
        self.llm_service.connect_to_llm()

        self.query_orchestrator = QueryOrchestrator(
            es_client=self.es_client, llm_service=self.llm_service
        )
        self.evaluation_orchestrator = EvaluationOrchestrator(
            llm_service=self.llm_service
        )
        self.feedback_orchestrator = FeedbackOrchestrator(db_manager=self.db_manager)
        self.cost_calculation_orchestrator = CostCalculationOrchestrator()

    def run(self, context, question):
        conversation_id = str(uuid.uuid4())

        # Process the query and get the response
        response, token_stats = self.query_orchestrator.process_query(context, question)

        # Evaluate the response
        evaluation_result, _ = self.evaluation_orchestrator.evaluate_response(
            question, response
        )

        # Calculate the cost
        cost = self.cost_calculation_orchestrator.calculate_cost(token_stats)

        # Save the conversation to the database
        self.db_manager.save_conversation(
            conversation_id=conversation_id, question=question, answer=response
        )

        return conversation_id, response, evaluation_result, cost
