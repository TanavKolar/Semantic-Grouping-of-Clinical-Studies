from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_similarity(model, query, dataset_embeddings):
    query_embedding = model.encode([query])
    similarity_scores = cosine_similarity(query_embedding.reshape(1, -1), dataset_embeddings)
    return similarity_scores.flatten()

def get_top_k_studies(similarity_scores, k=10):
    top_indices = np.argsort(similarity_scores)[-k:][::-1]
    return top_indices
