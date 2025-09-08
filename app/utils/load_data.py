"""
app/utils/load_data.py

loads the dataset for the Home Credit Risk Streamlit dashboard.
"""
import pandas as pd
import streamlit as st
import os

@st.cache_data
# -> pd.DataFrame — is a type hint
# It tells you (and tools like linters/IDEs) that the function is expected to return a pandas.DataFrame.
def load_data(file_path="data/raw/application_train.csv") -> pd.DataFrame:
    """Load data from a CSV file and cache the result."""

    # Problem with default (low_memory=True) -> Pandas processes the file in small chunks.
    # -> Each chunk is analyzed separately for data types.
    # low_memory=False -> Pandas reads the whole file into memory first.
    # Stick with low_memory=True (default) when:
    # You’re working with a massive CSV that doesn’t fit well in memory.
    base_dir = os.path.dirname(os.path.dirname(__file__)) # go up from utils to app root
    repo_root = os.path.dirname(base_dir)  # go up again → repo root
    abs_path = os.path.join(repo_root, file_path)
    df = pd.read_csv(abs_path,low_memory=False)

    # Normalization: strip accidental leading/trailing spaces in column names
    df.columns = [str(c).strip() for c in df.columns]

    #print(df.columns)
    return df

# dataset_columns = load_data()
# print(dataset_columns)
