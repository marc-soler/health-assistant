# health-assistant
RAG system dedicated to answering health-related questions based on scientific evidence.

## Project overview


## Dataset


## Technologies


## Preparation


## Running the application
### Database


### Docker-compose


### Locally (?)


### Docker


## Using the application
If API
### CLI

### requests

### CURL

If GUI
### Streamlit


## Code summary


## Experiments


## Monitoring



## Documentation
1. Kaggle CLI
[Documentation](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication)

2. Kaggle Dataset
[Dataset](https://www.kaggle.com/datasets/jpmiller/layoutlm)

Command to download dataset:
kaggle datasets download -d jpmiller/layoutlm -f medquad.csv && 
unzip medquad.csv.zip && 
rm medquad.csv.zip

3. OpenAI Key


4. Retrieval evaluation
MinSearch

ElasticSearch

embedding model choice based on performance (from [huggingface](https://huggingface.co/spaces/mteb/leaderboard)), max tokens (5692 longest answer), and dimensions, albeit truncated. [Sentence transformer](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)


Minsearch {'hit_rate': 0.8276, 'mrr': 0.47836746031746097}
ElasticSearch - Text {'hit_rate': 0.8064, 'mrr': 0.6198600000000003}
ElasticSearch - Vector {'hit_rate': 0.9184, 'mrr': 0.7592866666666654}

4.1 Optimization
- Use the Right Similarity Algorithm: usinc cosine already.
- Hierarchical Navigable Small World (HNSW) graphs, which Elasticsearch supports starting from version 7.3.

HNSW: This is an efficient ANN algorithm that provides a good trade-off between search speed and accuracy. You can enable it by setting "method": "hnsw" in your knn query. -> Almost no uplift compared to regular knn, will not apply.
- 
