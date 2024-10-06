FROM python:3.12-slim

WORKDIR /app

# Install Poetry
RUN pip install poetry

COPY data/medquad_embeddings.parquet data/medquad_embeddings.parquet
COPY ["pyproject.toml", "poetry.lock", "./"]

# Install dependencies using Poetry
RUN poetry install --no-dev --no-interaction --no-ansi

COPY health_assistant .

EXPOSE 8501

# Default command to run the application
CMD ["poetry", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]