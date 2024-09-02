# %%
import pandas as pd
from openai import OpenAI

client = OpenAI()

# %%
df = pd.read_parquet("../data/medquad_clean.parquet")
documents = df.to_dict(orient="records")

# %%
prompt_template = """
You emulate a user of our health assistant application.
Formulate 5 questions this user might ask based on a provided medical explanation.
Make the questions specific to this explanation.
The record should contain the answer to the questions, and the questions should
be complete and not too short. Use as fewer words as possible from the record. 

The record:

focus area: {focus_area}
question: {question}
answer: {answer}
source: {source}

Provide the output in parsable JSON without using code blocks:

{{"questions": ["question1", "question2", ..., "question5"]}}
""".strip()

# %%
prompt = prompt_template.format(**documents[0])  # type: ignore


# %%
def llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# %%
questions = llm(prompt)

# %%
import json

if questions is not None:
    json_questions = json.loads(questions)
    # do something with json_questions
else:
    # handle the case where questions is None
    print("No questions received")


# %%
def generate_questions(doc):
    prompt = prompt_template.format(**doc)

    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    )

    json_response = response.choices[0].message.content
    return json_response


# %%
## To avoid high API usage and costs, a random sample of 500 docs will be used
import random

documents_sample = random.sample(documents, 500)

# %%
from tqdm.auto import tqdm
from json import JSONDecodeError

results = {}
for doc in tqdm(documents_sample):
    doc_id = doc["id"]
    if doc_id in results:
        continue

    questions_raw = generate_questions(doc)
    if questions_raw is not None:
        try:
            questions = json.loads(questions_raw)
            results[doc_id] = questions["questions"]
        except JSONDecodeError:
            pass

# %%
final_results = []

for doc_id, questions in results.items():
    for q in questions:
        final_results.append((doc_id, q))

df_results = pd.DataFrame(final_results, columns=["id", "question"])

# %%
df_results.to_parquet("../data/ground_truth_retrieval.parquet", index=False)
