FROM python:3.12-slim

WORKDIR /health_assistant

# Install Poetry
RUN pip install poetry

COPY data/medquad.csv data/medquad.csv
COPY ["pyproject.toml", "poetry.lock", "./"]

# Install dependencies using Poetry
RUN poetry install --no-dev --no-interaction --no-ansi

COPY health_assistant health_assistant/.

EXPOSE 8501

# Default command to run the application
CMD ["poetry", "run", "streamlit", "run", "health_assistant/app.py", "--server.port=8501", "--server.address=0.0.0.0"]