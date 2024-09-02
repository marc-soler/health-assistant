import logging
from elasticsearch import Elasticsearch
from embedding_service import EmbeddingService


class SearchService:
    """
    SearchService is responsible for performing search queries on the Elasticsearch index.
    It uses an EmbeddingService to convert queries into vector representations before searching.
    """

    def __init__(
        self,
        es_client=None,
        index_name="health-questions-vector",
        embedding_service=None,
    ):
        """
        Initializes the SearchService with an Elasticsearch client, index name, and embedding service.

        Args:
            es_client (Elasticsearch, optional): An existing Elasticsearch client instance. Defaults to None.
            index_name (str, optional): The name of the Elasticsearch index to search. Defaults to "health-questions-vector".
            embedding_service (EmbeddingService, optional): An instance of EmbeddingService to generate query embeddings.
        """
        self.es_client = es_client or Elasticsearch("http://localhost:9200")
        self.index_name = index_name
        self.embedding_service = embedding_service or EmbeddingService()
        self.logger = logging.getLogger(__name__)

    def search(self, query, top_n=5):
        """
        Executes a search query on the Elasticsearch index by first embedding the query.

        Args:
            query (str): The search query to be embedded and executed.
            top_n (int, optional): The maximum number of search results to return. Defaults to 5.

        Returns:
            list: A list of search results, where each result is a dictionary.

        Raises:
            Exception: If there is an issue with the search query or the Elasticsearch client.
        """
        # Generate the embedding vector for the query
        query_vector = self.embedding_service.embed_query(query)

        search_body = {
            "field": "question_answer_vector",
            "query_vector": query_vector,
            "k": top_n,
            "num_candidates": 10000,
        }

        try:
            response = self.es_client.search(index=self.index_name, knn=search_body)
            self.logger.info(f"Search query executed successfully: {query}")
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            self.logger.error(f"Failed to execute search query: {query}. Error: {e}")
            raise RuntimeError(f"Failed to execute search query: {e}")

    def set_index_name(self, index_name):
        """
        Sets the name of the Elasticsearch index to search.

        Args:
            index_name (str): The name of the Elasticsearch index.
        """
        self.index_name = index_name
        self.logger.info(f"SearchService index name set to: {index_name}")

    def set_es_client(self, es_client):
        """
        Sets the Elasticsearch client instance to be used for searching.

        Args:
            es_client (Elasticsearch): The Elasticsearch client instance.
        """
        self.es_client = es_client
        self.logger.info("Elasticsearch client has been updated.")

    def set_embedding_service(self, embedding_service):
        """
        Sets the embedding service to be used for generating query embeddings.

        Args:
            embedding_service (EmbeddingService): The embedding service instance.
        """
        self.embedding_service = embedding_service
        self.logger.info("EmbeddingService has been updated.")
