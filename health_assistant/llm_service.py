import logging
import os
from openai import OpenAI


class LLMService:
    """
    LLMService is responsible for interacting with a Large Language Model (LLM).
    It handles the connection to the LLM API and sending queries to retrieve responses.
    """

    def __init__(self, logger=None):
        """
        Initializes the LLMService by retrieving the API key from environment variables and setting up the OpenAI client.
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        if not self.api_key:
            self.logger.error("OPENAI_API_KEY environment variable not set.")
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for LLMService."
            )

        self.openai_client = OpenAI()

        self.logger.info(
            "LLMService initialized with API key from environment variable."
        )

    def connect_to_llm(self):
        """
        Validates the connection to the LLM API using the provided API key.

        Raises:
            RuntimeError: If the connection to the LLM API fails.
        """
        try:
            # A simple test to ensure the connection is valid
            models = self.openai_client.models.list()
            self.logger.info(
                f"Successfully connected to the LLM API. Available models: {len(models.data)}"
            )
        except Exception as e:
            self.logger.error(f"Failed to connect to the LLM API: {e}")
            raise RuntimeError(f"Failed to connect to the LLM API: {e}")

    def query_llm(self, prompt, model="gpt-4o-mini"):
        """
        Sends a chat completion query to the LLM and retrieves a response.

        Args:
            prompt (str): The input prompt to send to the LLM.
            model (str, optional): The LLM model to use. Defaults to "gpt-4o-mini".

        Returns:
            tuple: A tuple containing the response from the LLM and the token usage statistics.

        Raises:
            RuntimeError: If the query to the LLM API fails.
        """
        try:
            self.logger.info(f"Sending query to LLM")
            response = self.openai_client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}]
            )

            answer = response.choices[0].message.content

            token_stats = {
                "prompt_tokens": response.usage.prompt_tokens,  # type: ignore
                "completion_tokens": response.usage.completion_tokens,  # type: ignore
                "total_tokens": response.usage.total_tokens,  # type: ignore
            }

            self.logger.info("Received response from LLM.")
            return answer, token_stats
        except Exception as e:
            self.logger.error(f"Failed to query the LLM: {e}")
            raise RuntimeError(f"Failed to query the LLM: {e}")

    def close_llm_client(self):
        """
        Closes the LLM client.
        """
        self.openai_client.close()
        self.logger.info("LLM client closed.")
