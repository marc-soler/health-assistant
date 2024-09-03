class PromptBuilder:
    """
    PromptBuilder is responsible for constructing and customizing prompts for LLM interactions.
    It encapsulates the logic required to build and format prompts based on input data.
    """

    def __init__(self, template=None):
        """
        Initializes the PromptBuilder with an optional template for prompts.

        Args:
            template (str, optional): A template string for the prompt. Defaults to None.
        """
        self.template = template or "Context: {context}\nQuestion: {question}\nAnswer:"

    def create_prompt_context(self, search_results):
        """
        Creates a context from search results to be used in the prompt.

        Args:
            search_results (list of dict): A list of dictionaries containing search results.
                                        Each dictionary should have keys 'answer', 'focus_area', and 'source'.

        Returns:
            dict: A dictionary containing the concatenated 'answer', 'focus_area', and 'source' from the search results.
                The keys of the dictionary are 'answer', 'focus_area', and 'source'.
        """
        answer = ""
        focus_area = ""
        source = ""

        for doc in search_results:
            answer += doc["answer"] + "\n\n"
            focus_area += doc["focus_area"] + "\n\n"
            source += doc["source"] + "\n\n"

        return {
            "answer": answer.strip(),
            "focus_area": focus_area.strip(),
            "source": source.strip(),
        }

    def build_prompt(self, **kwargs):
        """
        Constructs a prompt based on the provided keyword arguments, matching the fields in the template.

        Args:
            **kwargs: Arbitrary keyword arguments corresponding to the fields required by the template.

        Returns:
            str: The constructed prompt ready for use in LLM interaction.

        Raises:
            ValueError: If required fields for the template are missing.
        """
        try:
            prompt = self.template.format(**kwargs)
            return prompt
        except KeyError as e:
            missing_field = str(e).strip("'")
            raise ValueError(
                f"Missing required field for prompt template: {missing_field}"
            )

    def customize_prompt(self, custom_template):
        """
        Customizes the template used for building prompts.

        Args:
            custom_template (str): A new template string for the prompt.

        Example:
            custom_template = "Background: {context}\nQuery: {question}\nResponse:"
        """
        self.template = custom_template
