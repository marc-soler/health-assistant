FROM python:3.12-slim

WORKDIR /app

# Install Poetry
RUN pip install poetry

COPY data/medquad.csv data/medquad.csv
COPY ["pyproject.toml", "poetry.lock", "./"]

# Install dependencies using Poetry
RUN poetry install --no-dev --no-interaction --no-ansi

COPY health_assistant .

EXPOSE 5000

# Run the application
CMD ["poetry", "run", "gunicorn", "--bind", "0.0.0.0:5000", "app:app"]