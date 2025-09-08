"""
app/Home.py

Starting point/ Entrypoint for the Home Credit Risk Streamlit dashboard.
Main page: project intro, features, KPIs and dataset preview.
"""

from utils import preprocessing as pp
from utils import load_data as ld
from pathlib import Path
import pandas as pd
import streamlit as st
from utils.theme import apply_theme, get_palette
import time

# Applu color theme
apply_theme()
PALETTE = get_palette()

# ---- 
PROCESSED_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "full_cleaned.csv"
RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "application_train.csv"

def preprocess_and_save(raw_path: str, processed_csv_path: str):
    """
    Run the preprocessing pipeline once and cache the resulting DataFrame in Streamlit's cache.
    Returns (df, summary) where summary is the preprocessing summary if available.
    This function will not re-run unless inputs change or cache is cleared.
    """
    dataset_main = ld.load_data(raw_path)
    print(dataset_main.shape)

    # Validate required cols
    required_columns = ['DAYS_BIRTH','DAYS_EMPLOYED','AMT_ANNUITY','AMT_INCOME_TOTAL',
                        'AMT_CREDIT']

    missing_cols = pp.validate_columns(dataset_main.columns, required_columns)
    if missing_cols:
        raise RuntimeError(f"Missing required columns: {missing_cols}")

    missingness_report = pp.missingness_report(dataset_main, top_N=10)
    # print(missingness_report)

    # compress the dataset
    df = pp.optimize_dataframe(dataset_main)

    # Handle Missing Values
    df,summary = pp.handle_missing_value(df)
    # print(dataset_after_missing_vals.head())


    # New derived columns
    df = pp.create_derived_columns(df)
    # print(new_dataset.head())

    # Outlier handling
    df = pp.handle_outliers(df,['AMT_ANNUITY'],0.01,0.99)

    # Rare label grouping
    for col in ['ORGANIZATION_TYPE','NAME_TYPE_SUITE']:
        # safe checks
        if col in dataset_main.columns:
            df = pp.rare_label_encoder(df,col,threshold=0.01)

    # Define income bracket Low/Mid/High
    df = pp.income_bracket(df)
    print(df.shape)
    print(summary)

    # Save processed data (to both csv and pkl)
    pp.save_processed_data(df,processed_csv_path)
    return df, summary

if PROCESSED_PATH.exists():
    # If processed file exists on disk, load it (ld.load_data may use caching too)
    processed_df = ld.load_data(str(PROCESSED_PATH))
    # compute a lightweight summary on load (fast): missingness/top few cols
    try:
        summary = pp.missingness_report(processed_df, top_N=10)
    except Exception:
        print("Could not compute summary on loaded processed data.")
        summary = {
            "rows": int(processed_df.shape[0]),
            "cols": int(processed_df.shape[1])
        }
else:
    # Run the cached preprocessing function (first run will do heavy work; subsequent runs return cached df)
    with st.spinner("Running preprocessing — this runs only the first time..."):
        processed_df, summary = preprocess_and_save(str(RAW_PATH), str(PROCESSED_PATH))
        # small UX pause is optional; remove time.sleep in production
        time.sleep(0.5)
        msg = st.success("Preprocessing complete.")
        time.sleep(3)  
        msg.empty()

# Sidebar filters (shared logic)
filters = pp.get_global_filters(processed_df)
filtered_df = pp.apply_global_filters(processed_df, filters)

# st.write("Preview of filtered data:")
# st.dataframe(filtered_df.head(50))

# --- HEADER: Title, subtitle and badges ---
st.title("📈 Credit Default Risk — Interactive Dashboard")
st.markdown("**A compact, professional Streamlit app to explore credit default risk, correalations and affordability.**")
st.markdown(
    """
    <div>
      <small> Built with: Python • Pandas • Streamlit • Matplotlib/Seaborn. </small>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Two-column intro + KPIs ---
left, right = st.columns([2, 1])

with left:
    st.header("What this project does 🔍")
    st.markdown(
        """
        - **Explore** applicant-level data and portfolio health (defaults vs repaid).  
        - **Segment** borrowers by demographics, income, employment and housing.  
        - **Diagnose** affordability with DTI / LTI and highlight risky cohorts.  
        - **Quickly prototype** candidate underwriting rules (LTI caps, income floors).
        """
    )
    st.subheader("Key features ✨")
    st.markdown(
        """
        - Multi-page dashboard: Overview, Target segmentation, Demographics, Financial health, Correlations.  
        - Global sidebar filters (gender, education, family status, housing, age, income bracket, employment tenure).  
        - 10 KPIs + 10 charts per page (assignment requirement).  
        - Corporate warm color theme and consistent visuals.  
        """
    )
    st.markdown("**How to use**")
    st.markdown(
        """
        1. Use the _Global Filters_ in the sidebar to slice the dataset.  
        2. Navigate to pages (top-left) to explore sections.  
        3. Export charts/data as needed (future improvement).  
        """
    )

with right:
    st.header("Quick snapshot ⚡")
    # compute a few quick KPIs from processed_df
    total_applicants = len(processed_df)
    default_rate = processed_df["TARGET"].mean() if "TARGET" in processed_df.columns else float("nan")
    avg_income = processed_df["AMT_INCOME_TOTAL"].mean() if "AMT_INCOME_TOTAL" in processed_df.columns else float("nan")
    avg_credit = processed_df["AMT_CREDIT"].mean() if "AMT_CREDIT" in processed_df.columns else float("nan")

    st.metric("Total applicants", f"{total_applicants:,}")
    st.metric("Default rate", f"{100*default_rate:.2f}%")
    st.metric("Avg income", f"{avg_income:,.0f}")
    st.metric("Avg credit", f"{avg_credit:,.0f}")

# --- Dataset info & preview ---
st.markdown("---")
st.subheader("Dataset & processing info 🗂️")

proc_info_col1, proc_info_col2 = st.columns([2, 1])
with proc_info_col1:
    st.markdown("- **Rows:** {:,} &nbsp;&nbsp;•&nbsp;&nbsp; **Columns:** {:,}".format(processed_df.shape[0], processed_df.shape[1]))
    st.markdown("- **Preprocessing summary**:")
    with st.expander("Show preprocessing sample summary"):
        # Show the summary if available from earlier steps (best-effort)
        if 'summary' in locals():
            st.write(summary)
        else:
            st.write("Summary not available in this session. See preprocessing logs or rerun pipeline.")

st.markdown("---")

# --- Sidebar: also show short help / credits ---
st.sidebar.title("About")
st.sidebar.markdown("This dashboard was built for the Home Credit assignment. Use the pages menu to navigate through analyses.")
st.sidebar.markdown("**Credits:** ***Uditesh Jha***")

# --- Show a small preview of filtered data (using your global filters) ---
st.subheader("Preview of dataset (filtered by global sidebar filters)")
st.dataframe(filtered_df.head(20))

# --- Footer suggestions (non-blocking) ---
st.markdown(
    """
    ---
    **Next improvements (ideas):**  
    - Improve preprocessing and cache handling.
    - Add export buttons (download CSV / PNG).  
    - Add interactive Plotly charts for drill-down.  
    - Add a theme toggle (light/dark) and a professional logo.  
    """
)