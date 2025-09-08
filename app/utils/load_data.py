<<<<<<< HEAD
"""
app/utils/load_data.py

loads the dataset for the Home Credit Risk Streamlit dashboard.
"""
import pandas as pd
import streamlit as st

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
    df = pd.read_csv(file_path,low_memory=False)

    # Normalization: strip accidental leading/trailing spaces in column names
    df.columns = [str(c).strip() for c in df.columns]

    #print(df.columns)
    return df

# dataset_columns = load_data()
# print(dataset_columns)
=======
"""
app/utils/load_data.py

loads the dataset for the Home Credit Risk Streamlit dashboard.
"""
import pandas as pd
import streamlit as st

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
    df = pd.read_csv(file_path,low_memory=False)

    # Normalization: strip accidental leading/trailing spaces in column names
    df.columns = [str(c).strip() for c in df.columns]

    #print(df.columns)
    return df

# dataset_columns = load_data()
# print(dataset_columns)
>>>>>>> eed955b (created Home Credit Risk - Dashboard)
