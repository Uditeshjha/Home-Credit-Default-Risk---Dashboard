"""
app/utils/preprocessing.py

Preprocessing utilities/functions for the Home Credit Risk Streamlit dashboard.

"""
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from typing import List, Dict, Optional, Tuple

#########################################
# GENERAL FOR ALL FUNCTIONS
# Make the copy of dataframe and modify it


# -> pd.DataFrame — is a type hint
# It tells you (and tools like linters/IDEs) that the function is expected to return a pandas.DataFrame.
def validate_columns(df_cols, required_cols) -> pd.DataFrame:
    """
    Check if required columns exist in the DataFrame.
    Returns a list of missing columns (Empty list means All required cols present).
    """
    missing_columns = [c for c in required_cols if c not in df_cols]
    return missing_columns

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize numeric dtypes:
      - Integers → int8 / int16 / int32
      - Floats   → float16 / float32
    Returns a DataFrame with reduced memory usage.
    """
    before = df.memory_usage(deep=True).sum() / 1024**2
    optimized = df.copy()

    for col in optimized.columns:
        s = optimized[col]

        # --- Integers ---
        if pd.api.types.is_integer_dtype(s):
            vals = s.astype("int64")  # safe for bound checks
            if vals.min() >= np.iinfo(np.int8).min and vals.max() <= np.iinfo(np.int8).max:
                optimized[col] = s.astype("int8")
            elif vals.min() >= np.iinfo(np.int16).min and vals.max() <= np.iinfo(np.int16).max:
                optimized[col] = s.astype("int16")
            elif vals.min() >= np.iinfo(np.int32).min and vals.max() <= np.iinfo(np.int32).max:
                optimized[col] = s.astype("int32")
            # else: leave unchanged if too large

        # --- Floats ---
        elif pd.api.types.is_float_dtype(s):
            s64 = s.astype("float64")
            if np.allclose(s64, s64.astype("float16"), rtol=1e-03, atol=1e-06, equal_nan=True):
                optimized[col] = s64.astype("float16")
            else:
                optimized[col] = s64.astype("float32")

    after = optimized.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory: {before:.3f} MB → {after:.3f} MB ({(before - after) / before * 100:.2f}% reduction)")

    return optimized

def handle_missing_value(df,copy : bool = True) -> tuple[pd.DataFrame, dict[str]]: # type: ignore
    """
    Handle missing values:
    - Drop columns with missing_pct > drop_threshold
    - Impute numeric columns with median (default) or mean
    - Impute categorical columns with mode (default) or constant 'MISSING'

    Returns:
    new DataFrame (copy) and a dict with summary actions (dropped cols, imputed cols)

    """
    if copy:
        df = df.copy()
    summary = {"dropped_columns":[], "numeric_imputed":{}, "categorical_imputed":{}}

    # Drop columns with too many missing values
    # The reason we take .mean() after .isnull() is because it converts the True/False matrix into proportions 
    # of missing values per column.
    missing_pct = df.isnull().mean()
    # 60% missing values threshold
    drop_threshold = 0.6
    drop_cols = missing_pct[missing_pct > drop_threshold].index.tolist()
    if drop_cols:
        df = df.drop(columns=drop_cols)
        summary['dropped_columns'] = drop_cols
    
    # Numeric columns imputation
    # Considered few columns as of now
    cols_to_num_imputed = ['EXT_SOURCE_1','EXT_SOURCE_3']  
    for c in cols_to_num_imputed:
        if c in df.columns and df[c].isnull().any():
            val = df[c].median()
            df[c] = df[c].fillna(val)
            summary['numeric_imputed'][c] = val

    # Categorical columns imputation
    cols_to_cat_imputed = ['OCCUPATION_TYPE','NAME_TYPE_SUITE']
    for c in cols_to_cat_imputed:
        if c in df.columns and df[c].isnull().any():
            val = "MISSING"
            df[c] = df[c].fillna(val)
            summary['categorical_imputed'][c] = val
    
    return df, summary

def missingness_report(df,top_N ) -> pd.DataFrame:
    """
    Return a small DataFrame summarizing missingness.

    Columns:
      - column: column name
      - missing_pct: percent missing (0-100)
      - dtype: pandas dtype as string
      - unique_vals: number of unique values (including NaN)

    The result is sorted by missing_pct in descending order and trimmed to top_n rows.
    """
    missing_percentage = (df.isnull().mean() * 100).sort_values(ascending=False)
    report = pd.DataFrame({
        "missing_pct": missing_percentage,
        "data_type": df.dtypes.loc[missing_percentage.index].astype(str),
        "unique_vals": df.nunique(dropna=False).loc[missing_percentage.index]
    }).reset_index().rename(columns={"index": "column"})
    return report.head(top_N)

def create_derived_columns(df,copy: bool = True) -> pd.DataFrame:
    """
    Create and add derived columns used in the dashboard.

    Derived columns added (if required inputs exist):
      - AGE_YEARS           : floor((-DAYS_BIRTH) / 365)
      - EMPLOYMENT_YEARS    : floor((-DAYS_EMPLOYED) / 365) with sentinel handling
      - DTI                 : AMT_ANNUITY / AMT_INCOME_TOTAL (Debt-to-Income)
      - LTI                 : AMT_CREDIT / AMT_INCOME_TOTAL (Loan-to-Income)
      - ANNUITY_TO_CREDIT   : AMT_ANNUITY / AMT_CREDIT

    Notes:
      - If input column is missing, the output column will be pd.NA (nullable type).
      - Uses pandas nullable dtypes (Int64, Float64) where appropriate.
    """
    if copy:
        df = df.copy()

    # AGE_YEARS Column
    if 'DAYS_BIRTH' in df.columns:
        df['AGE_YEARS'] = np.floor(-safe_numeric(df['DAYS_BIRTH'])/365.25).astype('Int64')
    else:
        df['AGE_YEARS'] = pd.NA


    # EMPLOYMENT_YEARS Column
    if 'DAYS_EMPLOYED' in df.columns:
        # Sentinel value checks, more than 75 years is Sentinel value
        sentinel_mask = 27375
        if (df['DAYS_EMPLOYED'] < sentinel_mask).any():
            df['EMPLOYMENT_YEARS'] = np.floor(-safe_numeric(df['DAYS_EMPLOYED'])/365.25).astype('Int64')
    else:
        df['EMPLOYMENT_YEARS'] = pd.NA

    # DTI Column (Debt-to-Income)
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df['DTI'] = safe_divide(df['AMT_ANNUITY'],df['AMT_INCOME_TOTAL']).astype('Float64')
    else:
        df['DTI'] = pd.NA

    # LTI Column (Loan-to-Income)
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df['LTI'] = safe_divide(df['AMT_CREDIT'],df['AMT_INCOME_TOTAL']).astype('Float64')
    else:
        df['LTI'] = pd.NA
    
    # ANNUITY_TO_CREDIT Column
    if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
        df['ANNUITY_TO_CREDIT'] = safe_divide(df['AMT_ANNUITY'], df['AMT_CREDIT']).astype('Float64')
    else:
        df['ANNUITY_TO_CREDIT'] = pd.NA

    return df

def income_bracket(df, copy: bool = True) -> pd.DataFrame:
    """
    Define income brackets based on quartiles:
      - 'Low'  => values <= Q1 (25th percentile)
      - 'Mid'  => values > Q1 and <= Q3 (25-75th percentile)
      - 'High' => values > Q3 (top 25%)

    Stores the new column in df
    """
    if copy:
        df = df.copy()
    # errors="coerce": force invalid parsing into NaN, instead of failing.
    col_values = pd.to_numeric(df['AMT_INCOME_TOTAL'], errors='coerce')
    q1 = col_values.quantile(0.25)
    q3 = col_values.quantile(0.75)

    def bracket(val):
        if pd.isna(val):
            return pd.NA
        elif val <= q1:
            return "Low"
        elif val <= q3:
            return "Mid"
        else:
            return "High"
    new_column = "INCOME_BRACKET"
    df[new_column] = col_values.apply(bracket).astype("object")
    return df

def handle_outliers(df,cols,lower_pct,higher_pct,copy: bool = True) -> pd.DataFrame:
    """
    Clip numeric columns to the [lower_pct, upper_pct] quantile range.
    - cols: list of column names to clip
    - lower_pct, upper_pct: quantile bounds (0 < lower_pct < upper_pct < 1)
    - Returns a new DataFrame by default.
    """
    if copy:
        df = df.copy()

    for c in cols:
        if c in df.columns:
            lower_bound = df[c].quantile(lower_pct)
            upper_bound = df[c].quantile(higher_pct)
            df[c] = df[c].clip(lower=lower_bound, upper=upper_bound)
    return df

def rare_label_encoder(df, col,threshold, copy: bool = True) -> pd.DataFrame:
    """
    Purpose of this function: reduce cardinality for categorical variables by grouping rare categories into OTHER.
    """
    if copy:
        df = df.copy()

    if col not in df.columns:
        return df
    
    # normalize=False → gives raw counts
    # normalize=True  → gives proportions (0-1); fractions of the total/ gives relative frequencies
    # value_counts returns a Series indexed by the unique values in the column
    freq = df[col].value_counts(normalize=True, dropna=False)
    # index gives the category names
    to_replace = freq[freq  < threshold].index.tolist()
    if to_replace:
        df[col] = df[col].replace(to_replace, 'OTHER')
    return df 

#########################################
# Save all processed data to csv and pickle files
def save_processed_data(df, base_path) -> Dict[str, str]:
    """
    Save DataFrame to both CSV and PKL paths derived from base_path.
    Returns a dict with 'csv' and 'pkl' file paths.
    """
    p = Path(base_path)
    csv_path = str(p.with_suffix('.csv'))
    pkl_path = str(p.with_suffix('.pkl'))
    # pickel file is fast, keeps dtypes intact.
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    return {"csv": csv_path, "pkl": pkl_path}


#########################################
# Global Filters
def get_global_filters(df):
    """Build sidebar widgets and return filters dictionary"""
    gender_opts = sorted(df["CODE_GENDER"].dropna().unique()) if "CODE_GENDER" in df.columns else []
    educ_opts = sorted(df["NAME_EDUCATION_TYPE"].dropna().unique()) if "NAME_EDUCATION_TYPE" in df.columns else []
    fam_opts = sorted(df["NAME_FAMILY_STATUS"].dropna().unique()) if "NAME_FAMILY_STATUS" in df.columns else []
    house_opts = sorted(df["NAME_HOUSING_TYPE"].dropna().unique()) if "NAME_HOUSING_TYPE" in df.columns else []
    inc_opts = sorted(df["INCOME_BRACKET"].dropna().unique()) if "INCOME_BRACKET" in df.columns else []

    # Use explicit keys that match our filter names; widgets will maintain their values in st.session_state[key]
    filters = {}
    filters["CODE_GENDER"] = st.sidebar.multiselect(
        "Gender",
        options=gender_opts,
        default=st.session_state.get("CODE_GENDER", []),
        key="CODE_GENDER"
    )
    filters["NAME_EDUCATION_TYPE"] = st.sidebar.multiselect(
        "Education",
        options=educ_opts,
        default=st.session_state.get("NAME_EDUCATION_TYPE", []),
        key="NAME_EDUCATION_TYPE"
    )
    filters["NAME_FAMILY_STATUS"] = st.sidebar.multiselect(
        "Family Status",
        options=fam_opts,
        default=st.session_state.get("NAME_FAMILY_STATUS", []),
        key="NAME_FAMILY_STATUS"
    )
    filters["NAME_HOUSING_TYPE"] = st.sidebar.multiselect(
        "Housing",
        options=house_opts,
        default=st.session_state.get("NAME_HOUSING_TYPE", []),
        key="NAME_HOUSING_TYPE"
    )
    filters["INCOME_BRACKET"] = st.sidebar.multiselect(
        "Income Bracket",
        options=inc_opts,
        default=st.session_state.get("INCOME_BRACKET", []),
        key="INCOME_BRACKET"
    )

    # Sliders: give them explicit keys too and default to last saved or sensible range
    min_age = int(df["AGE_YEARS"].min()) if "AGE_YEARS" in df.columns else 18
    max_age = int(df["AGE_YEARS"].max()) if "AGE_YEARS" in df.columns else 100
    default_age = st.session_state.get("AGE_RANGE", (max(min_age, 18), min(max_age, 100)))
    filters["AGE_RANGE"] = st.sidebar.slider("Age Range", min_age, max_age, default_age, key="AGE_RANGE")

    min_emp = int(max(0, df["EMPLOYMENT_YEARS"].min())) if "EMPLOYMENT_YEARS" in df.columns else 0
    max_emp = int(max(0, df["EMPLOYMENT_YEARS"].max())) if "EMPLOYMENT_YEARS" in df.columns else 50
    default_emp = st.session_state.get("EMPLOYMENT_RANGE", (min_emp, min(max_emp, 50)))
    filters["EMPLOYMENT_RANGE"] = st.sidebar.slider("Employment Years", min_emp, max_emp, default_emp, key="EMPLOYMENT_RANGE")

    # Do NOT write filters back into st.session_state here — widgets update session_state automatically.
    return filters

def apply_global_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply global filters to a DataFrame.
    Expected filter keys (if present in dict and df):
      - CODE_GENDER: list
      - NAME_EDUCATION_TYPE: list
      - NAME_FAMILY_STATUS: list
      - NAME_HOUSING_TYPE: list
      - INCOME_BRACKET: list
      - AGE_RANGE: (min_age, max_age)
      - EMPLOYMENT_RANGE: (min_years, max_years)
    """
    fdf = df.copy()

    # All categorical filters
    for col in ["CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
                "NAME_HOUSING_TYPE", "INCOME_BRACKET"]:
        if col in filters and col in fdf.columns and filters[col]:
            fdf = fdf[fdf[col].isin(filters[col])]

    # AGE range
    if "AGE_RANGE" in filters and "AGE_YEARS" in fdf.columns:
        lo, hi = filters["AGE_RANGE"]
        fdf = fdf[(fdf["AGE_YEARS"] >= lo) & (fdf["AGE_YEARS"] <= hi)]

    # Employment range
    if "EMPLOYMENT_RANGE" in filters and "EMPLOYMENT_YEARS" in fdf.columns:
        lo, hi = filters["EMPLOYMENT_RANGE"]
        fdf = fdf[(fdf["EMPLOYMENT_YEARS"] >= lo) & (fdf["EMPLOYMENT_YEARS"] <= hi)]

    # drop=True: This parameter modifies the default behavior of reset_index(). 
    # When drop is set to True, the original index is not added as a new column to the DataFrame. It is simply dropped.
    return fdf.reset_index(drop=True)


#########################################
# Helper methods
def safe_numeric(num):
    """Convert to numeric, return pd.NA if conversion fails."""
    return pd.to_numeric(num, errors="coerce")

    
def safe_divide(numerator, denominator):
    """Perform safe division, return pd.NA if denominator is zero or pd.NA."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    # Replace 0 with NaN in denominator to avoid division by zero
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator