import os
import logging
import pandas as pd
import elasticsearch


class ElasticsearchIndexer:
    def __init__(self, logger=None, data_path=None, es_host="http://localhost:9200"):
        """
        Initializes the ElasticsearchIndexer with the data path and Elasticsearch host.
        """
        self.data_path = data_path or os.getenv(
            "DATA_PATH", "../data/medquad_embeddings.parquet"
        )
        self.es_client = elasticsearch.Elasticsearch(es_host)
        self.index_name = "health-questions-vector"
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def load_data(self):
        """
        Loads the data from the specified parquet file and returns it as a list of dictionaries.
        """
        try:
            df = pd.read_parquet(self.data_path)
            self.logger.info(
                f"Loaded data from {self.data_path} with {len(df)} records."
            )
            return df.to_dict(orient="records")
        except Exception as e:
            self.logger.error(f"Failed to load data from {self.data_path}: {e}")
            raise IOError(f"Failed to load data from {self.data_path}: {e}")

    def create_index(self):
        """
        Creates an Elasticsearch index with the specified settings and mappings.
        """
        es_vector_index_settings = {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "question": {"type": "text"},
                    "answer": {"type": "text"},
                    "source": {"type": "keyword"},
                    "focus_area": {"type": "keyword"},
                    "id": {"type": "keyword"},
                    "question_vector": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "answer_vector": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "question_answer_vector": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            },
        }

        try:
            self.es_client.indices.delete(
                index=self.index_name, ignore_unavailable=True
            )
            self.es_client.indices.create(
                index=self.index_name, body=es_vector_index_settings
            )
            self.logger.info(f"Created Elasticsearch index: {self.index_name}")
        except Exception as e:
            self.es_client.close()
            self.logger.error(f"Failed to create index {self.index_name}: {e}")
            raise RuntimeError(f"Failed to create index {self.index_name}: {e}")

    def index_documents(self, documents):
        """
        Indexes the provided documents into the Elasticsearch index.
        """
        for doc in documents:
            try:
                self.es_client.index(index=self.index_name, body=doc)  # type: ignore
            except Exception as e:
                self.logger.error(f"Failed to index document {doc['id']}: {e}")

    def load_and_index_data(self):
        """
        Loads data from the parquet file and indexes it into Elasticsearch.
        """
        documents = self.load_data()
        self.create_index()
        self.index_documents(documents)
        self.es_client.close()
        return self.es_client, self.index_name
