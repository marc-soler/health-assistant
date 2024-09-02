# %%
# RETRIEVAL EVALUATION
# %%
## DATA INGESTION
import pandas as pd

df = pd.read_parquet("../data/medquad_embeddings.parquet")
documents = df.to_dict(orient="records")
ground_truth_df = pd.read_parquet("../data/ground_truth_retrieval.parquet")
ground_truth = ground_truth_df.to_dict(orient="records")

# %%
### MINSEARCH
# !wget https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py
import minsearch

minsearch_index = minsearch.Index(
    text_fields=["question", "answer", "source", "focus_area"],
    keyword_fields=["id"],
)

minsearch_index.fit(documents)

# %%
### ELASTICSEARCH
import elasticsearch
from tqdm.auto import tqdm

es_client = elasticsearch.Elasticsearch("http://localhost:9200")

# %%
#### TEXT INDEX
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

for doc in tqdm(documents):
    doc = {
        "question": doc["question"],
        "answer": doc["answer"],
        "source": doc["source"],
        "focus_area": doc["focus_area"],
        "id": doc["id"],
    }
    es_client.index(index=index_name, body=doc)  # type: ignore

# %%
#### VECTOR INDEX
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
                "dims": 1024,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
}
index_name = "health-questions-vector"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=es_vector_index_settings)

for doc in tqdm(documents):
    es_client.index(index=index_name, body=doc)  # type: ignore

# %%
## OPENAI + SENTENCE TRANSFORMER
from openai import OpenAI
from sentence_transformers import SentenceTransformer

client = OpenAI()

embedding_model = SentenceTransformer(
    "Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True
)
embedding_model.max_seq_length = 1024
embedding_model.tokenizer.padding_side = "right"


# %%
## SEARCH FUNCTIONS
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


def search_elasticsearch_vector(query: str):
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


# %%
## METRICS
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
## EVALUATION FUNCTION
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
# %%
### MINSEARCH
minsearch_results = evaluate(ground_truth, lambda q: search_minsearch(q["question"]))
print(minsearch_results)
# {'hit_rate': 0.8276, 'mrr': 0.47836746031746097}

# %%
### ELASTICSEARCH - TEXT
elasticsearch_text_results = evaluate(
    ground_truth,
    lambda q: search_elasticsearch_text(q["question"]),
)
print(elasticsearch_text_results)
# {'hit_rate': 0.8064, 'mrr': 0.6198600000000003}

# %%
### ELASTICSEARCH - VECTOR
elasticsearch_vector_results = evaluate(
    ground_truth,
    lambda q: search_elasticsearch_vector(q["question"]),
)
print(elasticsearch_vector_results)
# {'hit_rate': 0.9184, 'mrr': 0.7592866666666654}


# %%
## QUERY OPTIMIZATION FOR ELASTICSEARCH VECTOR
# %%
### HNSW
es_vector_hnsw_index_settings = {
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
                "index_options": {"type": "hnsw", "m": 16, "ef_construction": 100},
            },
            "answer_vector": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine",
                "index_options": {"type": "hnsw", "m": 16, "ef_construction": 100},
            },
            "question_answer_vector": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine",
                "index_options": {"type": "hnsw", "m": 16, "ef_construction": 100},
            },
        }
    },
}
index_name = "health-questions-vector-hnsw"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=es_vector_hnsw_index_settings)

for doc in tqdm(documents):
    es_client.index(index=index_name, body=doc)  # type: ignore


# %%
def search_elasticsearch_vector_hnsw(query: str):
    vector = [t.tolist() for t in embedding_model.encode(query)]
    search_query = {
        "field": "question_answer_vector",
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
    }

    es_results = es_client.search(
        index="health-questions-vector-hnsw",
        knn=search_query,
    )

    return [hit["_source"] for hit in es_results["hits"]["hits"]]


elasticsearch_vector_hnsw_results = evaluate(
    ground_truth,
    lambda q: search_elasticsearch_vector_hnsw(q["question"]),
)
print(elasticsearch_vector_hnsw_results)
# {'hit_rate': 0.9184, 'mrr': 0.7593199999999987}
