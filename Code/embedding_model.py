from sentence_transformers import SentenceTransformer
import numpy as np

def initialize_model(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
    model = SentenceTransformer(model_name)
    return model

def load_embeddings():
    criteria_embeddings = np.load('embeddings/criteria_embeddings.npy')
    title_embeddings = np.load('embeddings/title_embeddings.npy')
    primary_embeddings = np.load('embeddings/primary_embeddings.npy')
    secondary_embeddings = np.load('embeddings/secondary_embeddings.npy')

    metadata_embeddings = (criteria_embeddings * 0.3 + primary_embeddings * 0.2 + secondary_embeddings * 0.1)
    enriched_embeddings = title_embeddings * 0.7 + metadata_embeddings

    return enriched_embeddings

def generate_embeddings(model, merged_df_pruned):
    title_embeddings = model.encode(merged_df_pruned['Study Title'].fillna(''), show_progress_bar=True)
    primary_embeddings = model.encode(merged_df_pruned['Primary Outcome Measures'].fillna(''), show_progress_bar=True)
    secondary_embeddings = model.encode(merged_df_pruned['Secondary Outcome Measures'].fillna(''), show_progress_bar=True)
    criteria_embeddings = model.encode(merged_df_pruned['criteria'].fillna(''), show_progress_bar=True)

    # Save embeddings as a .npy file
    np.save('embeddings/primary_embeddings.npy', primary_embeddings)
    np.save('embeddings/secondary_embeddings.npy', secondary_embeddings)
    np.save('embeddings/criteria_embeddings.npy', criteria_embeddings)
    np.save('embeddings/title_embeddings.npy', title_embeddings)


