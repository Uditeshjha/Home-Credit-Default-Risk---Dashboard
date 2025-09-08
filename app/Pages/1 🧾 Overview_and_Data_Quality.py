# pages/1_Overview_and_Data_Quality.py
"""
Page 1 — Overview & Data Quality (no local helper functions).

This page:
 - Loads processed CSV: data/processed/full_cleaned.csv (via utils.load_data for consistency)
 - Builds sidebar global filters via pp.get_global_filters(df)
 - Applies filters via pp.apply_global_filters(df, filters)
 - Shows 10 KPIs and 10 compact charts
 - Uses seaborn/matplotlib for visuals
 - Give few insights based on charts/KPIs and it is dynamic to filters

Notes:
 - Ensure data/processed/full_cleaned.csv exists (or change path if needed).
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import preprocessing as pp
from utils import load_data as ld

from utils.theme import apply_theme,  get_palette

# Applu color theme
apply_theme()
PALETTE = get_palette()

# Page header
st.title("Overview & Data Quality")
st.write("KPIs, data quality checks and core distributions. All visuals respond to the global filters in the sidebar.")

# Load the processed CSV
df_full = ld.load_data("../data/processed/full_cleaned.csv")


# Global filters
# pp.get_global_filters will create sidebar widgets and store selections in st.session_state.
filters = pp.get_global_filters(df_full)

# Apply filters consistently using your apply_global_filters
df = pp.apply_global_filters(df_full, filters)

# If no rows after filtering, show helpful message and stop early
if df.shape[0] == 0:
    st.warning("No rows match your current filters. Please broaden your filter selection.")
    st.stop()

# Helper method: small histogram drawing
# bins = 30 means Means: “divide the data range into 30 equal-width intervals.”
def _draw_small_hist(values, bins=30, xlabel=None,color = None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.hist(values.dropna(), bins=bins, edgecolor="k", alpha=0.7,color=color or PALETTE["navy"])
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

# KPIs (10) - computed on filtered df
st.subheader("Key Performance Indicators (KPIs)")

# compute KPI values defensively (check column exists)
total_applicants = len(df)
unique_customers = int(df["SK_ID_CURR"].nunique()) if "SK_ID_CURR" in df.columns else total_applicants
default_rate = float(df["TARGET"].mean()) if "TARGET" in df.columns else np.nan
avg_income = float(df["AMT_INCOME_TOTAL"].mean()) if "AMT_INCOME_TOTAL" in df.columns else np.nan
med_income = float(df["AMT_INCOME_TOTAL"].median()) if "AMT_INCOME_TOTAL" in df.columns else np.nan
avg_credit = float(df["AMT_CREDIT"].mean()) if "AMT_CREDIT" in df.columns else np.nan
med_credit = float(df["AMT_CREDIT"].median()) if "AMT_CREDIT" in df.columns else np.nan
avg_age = float(df["AGE_YEARS"].mean()) if "AGE_YEARS" in df.columns else np.nan
pct_female = float((df["CODE_GENDER"] == "F").mean()) if "CODE_GENDER" in df.columns else np.nan
n_cols_with_missing = int((df.isnull().mean() * 100 > 0).sum())

# Render KPIs in two rows and in 5 columns
cols = st.columns(5)
kpis = [
    ("Total applicants", f"{total_applicants:,}"),
    ("Unique customers", f"{unique_customers:,}"),
    ("Default rate", f"{100*default_rate:.2f}%" if not np.isnan(default_rate) else "N/A"),
    ("Avg income", f"{avg_income:,.0f}" if not np.isnan(avg_income) else "N/A"),
    ("Median income", f"{med_income:,.0f}" if not np.isnan(med_income) else "N/A"),
    ("Avg credit", f"{avg_credit:,.0f}" if not np.isnan(avg_credit) else "N/A"),
    ("Median credit", f"{med_credit:,.0f}" if not np.isnan(med_credit) else "N/A"),
    ("Avg age", f"{avg_age:.1f}" if not np.isnan(avg_age) else "N/A"),
    ("% Female", f"{100*pct_female:.2f}%" if not np.isnan(pct_female) else "N/A"),
    ("Columns with missing", f"{n_cols_with_missing}")
]

for i, (label, value) in enumerate(kpis):
    with cols[i % 5]:
        st.metric(label, value)

st.markdown("---")

# Data quality: missingness table + bar chart (top 10)
st.subheader("Data Quality — Missingness")
miss_report = pp.missingness_report(df, top_N=10)
st.dataframe(miss_report.style.format({"missing_pct": "{:.2f}"}), height=260)
st.markdown("---")

# horizontal bar chart for top 10 missing columns (simple)
top10 = miss_report.head(10).sort_values("missing_pct", ascending=True)
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(top10["column"], top10["missing_pct"], color="tab:orange")
ax.set_xlabel("Missingness (%)")
ax.set_title("Top 10 columns by missingness")
st.pyplot(fig)
plt.close(fig)

st.markdown("---")

# Ten charts (inline, simple)
st.subheader("Key Distributions & Charts (10)")

# Chart 1 & 2
c1, c2 = st.columns(2)


# -----------Code Breakdown-------------
# By default value_counts() drops NaNs; include them with dropna=False if you want a “Missing” bar.
# plt.subplots() is part of Matplotlib’s object-oriented API convenience factory ->
# It Creates a Figure object (fig) — the top-level container for the whole image and
# Creates one or more Axes objects (ax) — the area where data is drawn
# figsize is a tuple (width, height) measured in inches ->
# (4, 3) - about 400×300 pixels at 100 dpi.
# sns.barplot() is a Seaborn convenience that draws bars using Matplotlib under the hood.
# ax=ax → draw into the Axes instance you created ->
# This is what lets you combine multiple plots into one figure or control exactly where the plot appears.
# st.pyplot(fig) takes the Matplotlib Figure object, renders it to an image and displays that PNG in the Streamlit app.
# Matplotlib keeps objects in memory so plt.close(fig) releases that memory.
# ------------------------------------
with c1:
    st.markdown("**1) Target distribution**")
    if "TARGET" in df.columns:
        colors = [PALETTE["navy"], PALETTE["warm_orange"]]
        vc = df["TARGET"].value_counts().sort_index()
        # Map 0 → Repaid, 1 → Default
        label_map = {0: "Repaid", 1: "Default"}
        labels = [label_map.get(i, str(i)) for i in vc.index]
        fig, ax = plt.subplots(figsize=(3, 3))
        wedges, texts, autotexts = ax.pie(
            vc.values,
            labels=labels,
            autopct="%1.1f%%",   # show percentages
            startangle=90,       # rotate start for better look
            colors=colors[:len(vc.values)]
        )
        # Style labels properly
        for t in texts:       # outer labels (Repaid, Default)
            t.set(color="black", fontsize=8, weight="bold")
        for at in autotexts:  # percentages inside slices
            at.set(color="white", fontsize=9, weight="bold")
        ax.set(aspect="equal", title="Target distribution")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("TARGET not present.")

with c2:
    st.markdown("**2) Age distribution**")
    if "AGE_YEARS" in df.columns:
        _draw_small_hist(df["AGE_YEARS"].dropna(), bins=25)
    else:
        st.info("AGE_YEARS not present.")

# Chart 3 & 4
c3, c4 = st.columns(2)
with c3:
    # Income is skewed(unfair, inaccurate), so Log transform compresses large values and spreads small values → 
    # makes distribution more interpretable.
    st.markdown("**3) Income (log scale)**")
    if "AMT_INCOME_TOTAL" in df.columns:
        _draw_small_hist(np.log1p(df["AMT_INCOME_TOTAL"].dropna()), bins=30,color=PALETTE["warm_orange"])
    else:
        st.info("AMT_INCOME_TOTAL not present.")

with c4:
    st.markdown("**4) Credit amount**")
    if "AMT_CREDIT" in df.columns:
        _draw_small_hist(df["AMT_CREDIT"].dropna(), bins=30)
    else:
        st.info("AMT_CREDIT not present.")

# Chart 5 & 6
c5, c6 = st.columns(2)
with c5:
    st.markdown("**5) Annuity**")
    if "AMT_ANNUITY" in df.columns:
        _draw_small_hist(df["AMT_ANNUITY"].dropna(), bins=30)
    else:
        st.info("AMT_ANNUITY not present.")

with c6:
    st.markdown("**6) Debt-to-Income (DTI)**")
    if "DTI" in df.columns:
        _draw_small_hist(df["DTI"].dropna(), bins=30,color=PALETTE["warm_orange"])
    else:
        st.info("DTI not present.")

# Chart 7 & 8
c7, c8 = st.columns(2)
with c7:
    st.markdown("**7) Loan-to-Income (LTI)**")
    if "LTI" in df.columns:
        _draw_small_hist(df["LTI"].dropna(), bins=30,color=PALETTE["warm_orange"])
    else:
        st.info("LOAN_TO_INCOME not present.")

with c8:
    st.markdown("**8) Annuity/Credit**")
    if "ANNUITY_TO_CREDIT" in df.columns:
        _draw_small_hist(df["ANNUITY_TO_CREDIT"].dropna(), bins=30)
    else:
        st.info("ANNUITY_TO_CREDIT not present.")

# Chart 9 & 10: categories
c9, c10 = st.columns(2)
with c9:
    st.markdown("**9) Top education types**")
    if "NAME_EDUCATION_TYPE" in df.columns:
        top = df["NAME_EDUCATION_TYPE"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.barplot(y=top.index, x=top.values, ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("Education")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("NAME_EDUCATION_TYPE not present.")

with c10:
    st.markdown("**10) Housing types**")
    if "NAME_HOUSING_TYPE" in df.columns:
        top = df["NAME_HOUSING_TYPE"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.barplot(y=top.index, x=top.values, ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel("Housing")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("NAME_HOUSING_TYPE not present.")

st.markdown("---")

# key Insights
st.subheader("Quick Insights")
insights = []
if "TARGET" in df.columns:
    insights.append(f"- Filtered default rate: **{100*df['TARGET'].mean():.2f}%**.")
if not miss_report.empty:
    top_col = miss_report.iloc[0]
    insights.append(f"- Column with highest missingness: **{top_col['column']}** ({top_col['missing_pct']:.1f}% missing).")
if "AMT_INCOME_TOTAL" in df.columns:
    insights.append(f"- Income median: **{int(df['AMT_INCOME_TOTAL'].median()):,}**, mean: **{int(df['AMT_INCOME_TOTAL'].mean()):,}**.")

for s in insights:
    st.markdown(s)

st.caption("Tip: modify the global filters in the sidebar to slice the dataset. All visuals update accordingly.")
