# Clinical Trial Retrieval System
This project provides a pipeline for retrieving the most relevant clinical trials based on a user query. It integrates data preprocessing, embedding generation using PubMedBERT, similarity computation, and evaluation metrics such as Mean Reciprocal Rank (MRR) and Top-K accuracy.

## Embeddings Generation
The embeddings for the dataset fields canb be generated using PubMedBERT and stored in the embeddings/ folder as .npy files using the generate_embeddings function provided in the embedding_model module and called in main.py.

## How To Use
1) Run the main.py script to retrieve the top 10 relevant studies for a given query:
2) Enter the Study Title as the query
3) The top 10 studies and their similarity scores will be displayed.

## Acknowledgments
1) PubMedBERT: For generating semantic embeddings of clinical trial data.
2) Sentence Transformers: For handling embedding creation and cosine similarity computations.
