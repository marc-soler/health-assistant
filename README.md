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

embedding model choice based on performance (from [huggingface](https://huggingface.co/spaces/mteb/leaderboard)), max tokens (5692 longest answer), and dimensions, albeit truncated.