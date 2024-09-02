QUESTION_PROMPT_TEMPLATE = """
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

EVALUATION_PROMPT_TEMPLATE = """
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
