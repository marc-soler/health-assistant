# %%
# DATA INGESTION
import pandas as pd

df = pd.read_parquet("../data/medquad-embeddings.parquet")
documents = df.to_dict(orient="records")

# %%
## MINSEARCH
# !wget https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py
import minsearch

minsearch_index = minsearch.Index(
    text_fields=["question", "answer", "source", "focus_area"],
    keyword_fields=["id"],
)

minsearch_index.fit(documents)

# %%
## ELASTICSEARCH
import elasticsearch
from tqdm.auto import tqdm

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
# RAG FLOW
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


def build_prompt_first_version(query, search_results):
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
def rag(
    query,
    search_function,
    prompt_builder,
    model="gpt-4o-mini",
):
    search_results = search_function(query)
    prompt = prompt_builder(query, search_results)
    answer = llm(prompt, model=model)
    return answer


# %%
## TESTS
# %%
### MINSEARCH
question = "A new mole has appeared in my skin. What can this mean?"
answer = rag(question, search_minsearch, build_prompt_first_version)
print(answer)

# %%
### ELASTICSEARCH - TEXT
question = "A new mole has appeared in my skin. What can this mean?"
answer = rag(question, search_elasticsearch_text, build_prompt_first_version)
print(answer)

# %%
### ELASTICSEARCH - VECTOR
question = "A new mole has appeared in my skin. What can this mean?"
answer = rag(question, search_elasticsearch_vector, build_prompt_first_version)
print(answer)

# %%
# RAG EVALUATION
# %%
ground_truth_df = pd.read_parquet("../data/ground-truth-retrieval.parquet")
ground_truth = ground_truth_df.to_dict(orient="records")

# %%
## EVALUATION PROMPTS
prompt1_template = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to analyze the relevance of the generated answer compared to the original answer provided.
Based on the relevance and similarity of the generated answer to the original answer, you will classify
it as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Original Answer: {answer_orig}
Generated Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the original
answer and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()

prompt2_template = """
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

# %%
import json

# %%
df_sample = ground_truth_df.sample(n=200, random_state=1)
sample = df_sample.to_dict(orient="records")


# %%
def evaluate_rag(prompt_builder, eval_prompt_version):
    evaluations = []

    for record in tqdm(sample):
        question = record["question"]
        answer_orig = documents[record["id"]]
        answer_llm = rag(question, search_elasticsearch_vector, prompt_builder)

        if eval_prompt_version == 1:
            evaluation_prompt = prompt1_template.format(
                answer_orig=answer_orig, question=question, answer_llm=answer_llm
            )
        elif eval_prompt_version == 2:
            evaluation_prompt = prompt2_template.format(
                question=question, answer_llm=answer_llm
            )

        evaluation = llm(evaluation_prompt)

        if evaluation is not None:
            evaluation = json.loads(evaluation)
            evaluations.append((record, answer_llm, evaluation))
        else:
            # Handle the case where evaluation is None
            print("No evaluation received")

    df_eval = pd.DataFrame(evaluations, columns=["record", "answer", "evaluation"])

    df_eval["id"] = df_eval.record.apply(lambda d: d["id"])
    df_eval["question"] = df_eval.record.apply(lambda d: d["question"])

    df_eval["relevance"] = df_eval.evaluation.apply(lambda d: d["Relevance"])
    df_eval["explanation"] = df_eval.evaluation.apply(lambda d: d["Explanation"])

    del df_eval["record"]
    del df_eval["evaluation"]

    return df_eval


# %%
### PROMPT FIRST VERSION, COMPARING LLM ANSWER TO ORIGINAL ANSWER
df_eval_v1_q_a_a = evaluate_rag(build_prompt_first_version, eval_prompt_version=1)
df_eval_v1_q_a_a.relevance.value_counts(normalize=True)
df_eval_v1_q_a_a[df_eval_v1_q_a_a.relevance == "NON_RELEVANT"]
# relevance
# RELEVANT           0.515
# PARTLY_RELEVANT    0.475
# NON_RELEVANT       0.010

# This evaluation method seems to be a lot more sensitive than simply compaing question to answer.

# %%
df_eval_v1_q_a_a.to_parquet(
    "../data/rag-eval-gpt-4o-mini-prompt1-a-q-a.parquet", index=False
)

# %%
### PROMPT FIRST VERSION, COMPARING LLM ANSWER TO QUESTION
df_eval_v1_q_a = evaluate_rag(build_prompt_first_version, eval_prompt_version=2)
df_eval_v1_q_a.relevance.value_counts(normalize=True)
df_eval_v1_q_a[df_eval_v1_q_a.relevance == "NON_RELEVANT"]
# relevance
# RELEVANT           0.92
# PARTLY_RELEVANT    0.08

# %%
df_eval_v1_q_a.to_parquet(
    "../data/rag-eval-gpt-4o-mini-prompt1-a-q.parquet", index=False
)

# %%
## TESTING OTHER PROMPTS
prompt_template_improved = """
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


def build_prompt_improved(query, search_results):
    answer = ""
    focus_area = ""
    source = ""

    for doc in search_results:
        answer += doc["answer"] + "\n\n"
        focus_area += doc["focus_area"] + "\n\n"
        source += doc["source"] + "\n\n"

    prompt = prompt_template_improved.format(
        question=query, answer=answer, focus_area=focus_area, source=source
    ).strip()
    return prompt


# %%
### PROMPT IMPROVED VERSION, COMPARING LLM ANSWER TO ORIGINAL ANSWER
df_eval_v2_q_a_a = evaluate_rag(build_prompt_improved, eval_prompt_version=1)
df_eval_v2_q_a_a.relevance.value_counts(normalize=True)
df_eval_v2_q_a_a[df_eval_v2_q_a_a.relevance == "NON_RELEVANT"]
# relevance
# PARTLY_RELEVANT    0.51
# RELEVANT           0.46
# NON_RELEVANT       0.03

# %%
df_eval_v2_q_a_a.to_parquet(
    "../data/rag-eval-gpt-4o-mini-prompt2-a-q-a.parquet", index=False
)

# %%
### PROMPT IMPROVED VERSION, COMPARING LLM ANSWER TO QUESTION
df_eval_v2_q_a = evaluate_rag(build_prompt_improved, eval_prompt_version=2)
df_eval_v2_q_a.relevance.value_counts(normalize=True)
df_eval_v2_q_a[df_eval_v2_q_a.relevance == "NON_RELEVANT"]
# relevance
# RELEVANT           0.925
# PARTLY_RELEVANT    0.065
# NON_RELEVANT       0.010

# Original prompt doesn't yield non-relevant answers, whereas this does.
# However, non relevant answers are due to explicit instructions to state lack of context.
# In fact, the original prompt does not output any meaningful answer for the 2 questions that are non relevant.
# The honesty and transparency of the improved prompt ismuch more valued, especially considering it's providing medical advise.
# This prompt seems to reduce the appearance of hallucinations.

# %%
df_eval_v2_q_a.to_parquet(
    "../data/rag-eval-gpt-4o-mini-prompt2-a-q.parquet", index=False
)
