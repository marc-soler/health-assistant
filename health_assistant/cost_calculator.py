import logging


class CostCalculator:
    """
    CostCalculator is responsible for calculating the cost associated with using the OpenAI API.
    Specifically, it calculates the cost for the 'gpt-4o-mini' model.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_openai_cost(self, tokens):
        """
        Calculates the cost of using the OpenAI API for the 'gpt-4o-mini' model.

        Args:
            tokens (dict): A dictionary containing 'prompt_tokens' and 'completion_tokens'.

        Returns:
            float: The calculated OpenAI API cost.
        """
        try:
            openai_cost = (
                tokens["prompt_tokens"] * 0.00015 + tokens["completion_tokens"] * 0.0006
            ) / 1000
            self.logger.info(f"Calculated OpenAI cost: {openai_cost:.6f}")
            return openai_cost
        except KeyError as e:
            self.logger.error(f"Token data missing required field: {e}")
            return 0.0
