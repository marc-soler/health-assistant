from logging_setup import LoggerSetup
from elasticsearch_indexer import ElasticsearchIndexer
from prompt_builder import PromptBuilder
import prompt_templates as templates
from health_assistant.search_service import SearchService


if __name__ == "__main__":
    logger_setup = LoggerSetup()
    logger = logger_setup.get_logger()

    indexer = ElasticsearchIndexer()
    es_client, index_name = indexer.load_and_index_data()

    question_prompt_builder = PromptBuilder(templates.QUESTION_PROMPT_TEMPLATE)
    question_prompt = question_prompt_builder.build_prompt(
        context=context, question=question
    )

    evaliuation_prompt_builder = PromptBuilder(templates.EVALUATION_PROMPT_TEMPLATE)
    evaliuation_prompt = question_prompt_builder.build_prompt(
        context=context, question=question
    )

    search_service = SearchService(es_client=es_client)
    results = search_service.search(
        query="What are the side effects of hypertension medication?"
    )
