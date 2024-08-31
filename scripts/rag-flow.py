# %%
# INGESTION
# %%
import pandas as pd

# %%
df = pd.read_csv("../data/medquad-clean.csv")
documents = df.to_dict(orient="records")

# %%
## MINSEARCH
# !wget https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py
import minsearch

# %%
minsearch_index = minsearch.Index(
    text_fields=["question", "answer", "source", "focus_area"],
    keyword_fields=["id"],
)

minsearch_index.fit(documents)

# %%
## ELASTICSEARCH
import elasticsearch

es_client = elasticsearch.Elasticsearch("http://localhost:9200")

# %%
### TEXT INDEX
es_text_index_settings = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "question": {"type": "text"},
            "answer": {"type": "text"},
            "source": {"type": "keyword"},
            "focus_area": {"type": "keyword"},
            "id": {"type": "keyword"},
        }
    },
}
index_name = "health-questions-text"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=es_text_index_settings)

for doc in documents:
    es_client.index(index=index_name, body=doc)  # type: ignore

# %%
### VECTOR INDEX
es_vector_index_settings = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "question": {"type": "text"},
            "answer": {"type": "text"},
            "source": {"type": "keyword"},
            "focus_area": {"type": "keyword"},
            "id": {"type": "keyword"},
            "question_vector": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine",
            },
            "answer_vector": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine",
            },
            "question_answer_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": 1024,
                "similarity": "cosine",
            },
        }
    },
}
index_name = "health-questions-vector"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=es_vector_index_settings)

# for doc in documents_vectors:
#     es_client.index(index=index_name, body=doc)

# %%
# RAG FLOW
# %%
## OPENAI CLIENT
from openai import OpenAI

client = OpenAI()

# %%
## SEARCH FUNCTIONS
from typing import Dict


def search_minsearch(query: str):
    boost = {}

    results_minsearch = minsearch_index.search(
        query=query, filter_dict={}, boost_dict=boost, num_results=10
    )
    return results_minsearch


def search_elasticsearch_text(query: str):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question", "answer", "focus_area"],
                        "type": "best_fields",
                    }
                },
            }
        },
    }
    results_es = es_client.search(index="health-questions-text", body=search_query)
    return [i["_source"] for i in results_es["hits"]["hits"]]


def search_elasticsearch_vector(query, vector):
    vector = (
        client.embeddings.create(
            input=[query], model="text-embedding-3-small", dimensions=1024
        )
        .data[0]
        .embedding
    )
    knn = {
        "field": "question_answer_vector",
        "query_vector": vector,
        "k": 5,
        "num_candidates": 15000,
    }

    search_query = {
        "knn": knn,
        "_source": ["question", "answer", "source", "focus_area", "id"],
    }

    es_results = es_client.search(index="health-questions-vector", body=search_query)

    return [hit["_source"] for hit in es_results["hits"]["hits"]]


# %%
## PROMPT BUILDER
prompt_template = """
You're a health professional. Answer the QUESTION based on the CONTEXT from our exercises database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

entry_template = """
question: {question}
answer: {answer}
source: {source}
focus_area: {focus_area}
""".strip()


def build_prompt(query, search_results):
    context = ""

    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


# %%
## LLM ANSWER
def llm(prompt, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# %%
## RAG FLOW
def rag(query, search_function, model="gpt-4o-mini"):
    search_results = search_function(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, model=model)
    return answer


# %%
question = "A new mole has appeared in my skin. What can this mean?"
answer = rag(question, search_elasticsearch_text)
print(answer)

# %%
# RETRIEVAL EVALUATION
ground_truth_df = pd.read_csv("../data/ground-truth.csv")
ground_truth = ground_truth_df.to_dict(orient="records")


# %%
def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)


def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)


# %%
from tqdm.auto import tqdm


def minsearch_search(query):
    boost = {}

    results = minsearch_index.search(
        query=query, filter_dict={}, boost_dict=boost, num_results=10
    )

    return results


def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q["id"]
        results = search_function(q)
        relevance = [d["id"] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }


# %%
## RESULTS
misnearch_results = evaluate(ground_truth, lambda q: minsearch_search(q["question"]))
