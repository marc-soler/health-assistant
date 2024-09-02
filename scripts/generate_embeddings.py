# %%
import pandas as pd

df = pd.read_parquet("../data/medquad_clean.parquet")
docs = df.to_dict(orient="records")

# %%
# MAX TOKENS IN ANSWERS
import tiktoken


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


max_tokens = 0
for doc in docs:
    max_tokens = max(max_tokens, num_tokens_from_string(doc["answer"]))
print(max_tokens)
# 5692

# %%
## OPEN SOURCE MODEL
# model choice based on performance,
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
model.max_seq_length = 1024
model.tokenizer.padding_side = "right"

# %%
# USING PARALLELIZATION
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm


def process_doc(doc):
    question = doc["question"]
    answer = doc["answer"]
    qa = question + " " + answer

    # Encoding the question, answer, and combined question+answer
    question_vector = model.encode(question)
    answer_vector = model.encode(answer)
    question_answer_vector = model.encode(qa)

    # Return the result so we can update docs later
    return {
        "id": doc["id"],
        "question_vector": question_vector.tolist(),  # type: ignore
        "answer_vector": answer_vector.tolist(),  # type: ignore
        "question_answer_vector": question_answer_vector.tolist(),  # type: ignore
    }


# Number of threads
num_threads = 8  # Adjust according to your system's capabilities

# Use ThreadPoolExecutor to parallelize the encoding process
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Map the processing function to the docs
    futures = {executor.submit(process_doc, doc): doc for doc in docs}

    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        doc_id = result["id"]

        # Update the docs dictionary with the encoded vectors
        docs[doc_id]["question_vector"] = result["question_vector"]
        docs[doc_id]["answer_vector"] = result["answer_vector"]
        docs[doc_id]["question_answer_vector"] = result["question_answer_vector"]

# %%
# SAVE TO CSV
df = pd.DataFrame(docs)
df.to_parquet("../data/medquad_embeddings.parquet", index=False)
