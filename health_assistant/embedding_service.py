import logging
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    EmbeddingService is responsible for generating embeddings from text queries using
    a pre-trained sentence transformer model.
    """

    def __init__(self, logger=None, model_name="Alibaba-NLP/gte-large-en-v1.5"):
        """
        Initializes the EmbeddingService with a specific sentence transformer model.

        Args:
            model_name (str): The name of the pre-trained model to load.
        """
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model.max_seq_length = 1024
        self.model.tokenizer.padding_side = "right"
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def embed_query(self, query):
        """
        Generates an embedding vector for the provided query string.

        Args:
            query (str): The text query to embed.

        Returns:
            list: A list of vectors representing the query embedding.
        """
        try:
            self.logger.info(f"Embedding query: {query}")
            embedded_query = self.model.encode(query)
            self.logger.info(f"Embedded query: {embedded_query}")
            embedded_query_list = [t.tolist() for t in embedded_query]
            return embedded_query_list
        except Exception as e:
            self.logger.error(f"Failed to embed query: {query}. Error: {e}")
            raise RuntimeError(f"Failed to embed query: {query}. Error: {e}")
