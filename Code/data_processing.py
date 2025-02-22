import pandas as pd
import numpy as np


def load_data(trials_file, criteria_file):
    trials_df = pd.read_csv(trials_file)
    eligibilities_df = pd.read_csv(criteria_file, sep = '|')
    merged_df = pd.merge(trials_df, eligibilities_df, left_on='NCT Number', right_on='nct_id', how='inner')
    # List of columns to keep
    columns_to_keep = ['NCT Number', 'Study Title', 'Primary Outcome Measures', 'Secondary Outcome Measures', 'criteria']

    # Retain only the specified columns
    merged_df_pruned = merged_df.loc[:, columns_to_keep]

    return merged_df_pruned


