import pandas as pd
from data_processing import load_data
from embedding_model import initialize_model, load_embeddings
from similarity_computation import compute_similarity, get_top_k_studies
from evaluation import evaluate_model

#function to retrieve top 10 matches
def display_top_matches(top_indices):
    print("Most similar trials:")
    print(merged_df_pruned.iloc[top_indices][['NCT Number','Study Title']])

# File paths for the data files
TRIALS_FILE = "data/usecase_1_.csv"
CRITERIA_FILE = "data/eligibilities.txt"

# Constants
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
K = 10

merged_df_pruned = load_data(TRIALS_FILE, CRITERIA_FILE)

model = initialize_model(MODEL_NAME)

#generate embeddings which will get stored in the embeddings folder
generate_embeddings(model, merged_df_pruned)

embeddings = load_embeddings()

print("Enter the Query Clinical Trial Study Title: ")
query = input()

#generating similarity scores
similarity_scores = compute_similarity(model, query, embeddings)
top_indices = get_top_k_studies(similarity_scores, 10)
#retrieving top 10 matches
display_top_matches(top_indices)



# Code to evaluate using Top-k and MRR
'''sample_size = 1000  # Adjust the sample size as needed
results = evaluate_model(model, merged_df_pruned, sample_size=sample_size, top_k=K)

print(f"Top-10 Accuracy (Random Sample): {results['Top-K Accuracy']}")
print(f"Mean Reciprocal Rank (MRR, Random Sample): {results['MRR']}")'''