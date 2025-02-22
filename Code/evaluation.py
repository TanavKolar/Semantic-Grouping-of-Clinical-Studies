import random

# Function to evaluate the model on a random sample
def evaluate_model(model, dataset, sample_size=100, top_k=10):
    # Ensure the sample size is within the dataset size
    sample_size = min(sample_size, len(dataset))
    
    # Randomly sample entries from the dataset
    sample_indices = random.sample(range(len(dataset)), sample_size)
    sampled_dataset = dataset.iloc[sample_indices]
    
    top_k_hits = 0
    reciprocal_ranks = []

    for index, row in sampled_dataset.iterrows():
        query = row['Study Title']
        correct_trial_id = row['NCT Number']

        # Generate similarity scores for the query
        similarities = similarity_score_function(query)

        # Get the top-K indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Find the rank of the correct trial
        if index in top_indices:
            correct_rank = np.where(top_indices == index)[0][0] + 1
        else:
            correct_rank = len(similarities) + 1  # Assign a rank outside the range
        
        # Check if correct trial is in top-K
        if correct_rank <= top_k:
            top_k_hits += 1

        # Compute reciprocal rank
        reciprocal_ranks.append(1 / correct_rank)

    # Calculate metrics
    top_k_accuracy = top_k_hits / sample_size
    mrr = sum(reciprocal_ranks) / sample_size

    return {"Top-K Accuracy": top_k_accuracy, "MRR": mrr}
