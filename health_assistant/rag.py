import json
from time import time
import logging
from logging.config import dictConfig
import sys
from openai import OpenAI
from elasticsearch_indexer import ElasticsearchIndexer
from sentence_transformers import SentenceTransformer


def setup_logging():
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s %(levelname)s %(message)s",
            },
            "json": {
                "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "json",  # Change to 'default' if not using JSON
            },
            "error_console": {
                "class": "logging.StreamHandler",
                "stream": sys.stderr,
                "formatter": "json",  # Change to 'default' if not using JSON
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "error_console"],
        },
        "loggers": {
            "__main__": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False,
            },
        },
    }
    dictConfig(logging_config)


setup_logging()
logger = logging.getLogger(__name__)

indexer = ElasticsearchIndexer()
es_client, index_name = indexer.load_and_index_data()

embedding_model = SentenceTransformer(
    "Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True
)
embedding_model.max_seq_length = 1024
embedding_model.tokenizer.padding_side = "right"
openai_client = OpenAI()


def search_elasticsearch(query: str):
    vector = [t.tolist() for t in embedding_model.encode(query)]
    search_query = {
        "field": "question_answer_vector",
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
    }

    es_results = es_client.search(
        index="health-questions-vector",
        knn=search_query,
    )

    return [hit["_source"] for hit in es_results["hits"]["hits"]]


question_prompt_template = """
Role: You are a knowledgeable and experienced health professional.

Task: Provide a clear and concise answer to the QUESTION using only the information provided in the CONTEXT. Do not include any information outside of what is given.

Instructions:
- Base your answer strictly on the CONTEXT provided.
- If the CONTEXT does not contain sufficient information to fully answer the QUESTION, explicitly state that based on the CONTEXT.
- Maintain a professional tone and ensure the information is medically accurate.

QUESTION: {question}

CONTEXT:
- Answers from Medical Databases:
{answer}
- Focus Areas:
{focus_area}
- Sources:
{source}
""".strip()


def build_prompt(query, search_results):
    answer = ""
    focus_area = ""
    source = ""

    for doc in search_results:
        answer += doc["answer"] + "\n\n"
        focus_area += doc["focus_area"] + "\n\n"
        source += doc["source"] + "\n\n"

    prompt = question_prompt_template.format(
        question=query, answer=answer, focus_area=focus_area, source=source
    ).strip()
    return prompt


def llm(prompt, model="gpt-4o-mini"):
    response = openai_client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    token_stats = {
        "prompt_tokens": response.usage.prompt_tokens,  # type: ignore
        "completion_tokens": response.usage.completion_tokens,  # type: ignore
        "total_tokens": response.usage.total_tokens,  # type: ignore
    }

    return answer, token_stats


evaluation_prompt_template = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


def evaluate_relevance(question, answer):
    prompt = evaluation_prompt_template.format(question=question, answer=answer)
    evaluation, tokens = llm(prompt, model="gpt-4o-mini")

    if evaluation is not None:
        try:
            json_eval = json.loads(evaluation)
            return json_eval, tokens
        except json.JSONDecodeError:
            result = {
                "Relevance": "UNKNOWN",
                "Explanation": "Failed to parse evaluation",
            }
            return result, tokens
    else:
        result = {"Relevance": "UNKNOWN", "Explanation": "No evaluation received"}
        return result, tokens


def calculate_openai_cost(model, tokens):
    openai_cost = 0

    if model == "gpt-4o-mini":
        openai_cost = (
            tokens["prompt_tokens"] * 0.00015 + tokens["completion_tokens"] * 0.0006
        ) / 1000
    else:
        print("Model not recognized. OpenAI cost calculation failed.")

    return openai_cost


def rag(query, model="gpt-4o-mini"):
    t0 = time()

    search_results = search_elasticsearch(query)
    prompt = build_prompt(query, search_results)
    answer, token_stats = llm(prompt, model=model)

    relevance, rel_token_stats = evaluate_relevance(query, answer)

    t1 = time()
    took = t1 - t0

    openai_cost_rag = calculate_openai_cost(model, token_stats)
    openai_cost_eval = calculate_openai_cost(model, rel_token_stats)

    openai_cost = openai_cost_rag + openai_cost_eval

    answer_data = {
        "answer": answer,
        "model_used": model,
        "response_time": took,
        "relevance": relevance.get("Relevance", "UNKNOWN"),
        "relevance_explanation": relevance.get(
            "Explanation", "Failed to parse evaluation"
        ),
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
        "eval_prompt_tokens": rel_token_stats["prompt_tokens"],
        "eval_completion_tokens": rel_token_stats["completion_tokens"],
        "eval_total_tokens": rel_token_stats["total_tokens"],
        "openai_cost": openai_cost,
    }

    return answer_data
