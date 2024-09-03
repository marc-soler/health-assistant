FROM python:3.12-slim

WORKDIR /app

# Install Poetry
RUN pip install poetry

COPY data/medquad.csv data/medquad.csv
COPY ["pyproject.toml", "poetry.lock", "./"]

# Install dependencies using Poetry
RUN poetry install --no-dev --no-interaction --no-ansi

COPY health_assistant .

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8501

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command to run the application
CMD ["poetry", "run", "streamlit", "run", "health_assistant/app.py", "--server.port=8501", "--server.address=0.0.0.0"]