# pages/5_Correlations_Drivers.py
"""
Page 5 — Correlations, Drivers & Interactive Slice-and-Dice

Purpose: What drives default? Combine correlation views with interactive, filterable charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_data as ld
from utils import preprocessing as pp
from utils.theme import apply_theme, get_palette

apply_theme()
PALETTE = get_palette()

st.title("Correlations, Drivers & Interactive Slice-and-Dice")
st.write("Explore numeric correlations to TARGET and test candidate rules. All visuals respond to the global filters.")

# Load & filter
df_full = ld.load_data("data/processed/full_cleaned.csv")
filters = pp.get_global_filters(df_full)
df = pp.apply_global_filters(df_full, filters)

if df.shape[0] == 0:
    st.warning("No rows after applying filters. Please broaden the global filters.")
    st.stop()

# Prepare numeric columns for correlation
# choose a focused numeric set (avoid IDs)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# remove obvious ID-like columns if present
for c in ["SK_ID_CURR", "SK_ID_PREV"]:
    if c in numeric_cols:
        numeric_cols.remove(c)

# Ensure TARGET included
if "TARGET" not in numeric_cols and "TARGET" in df.columns:
    numeric_cols.append("TARGET")

# small helper to safe-get correlations
corr_matrix = df[numeric_cols].corr() if len(numeric_cols) > 1 else pd.DataFrame()
target_corr = corr_matrix["TARGET"].drop("TARGET") if ("TARGET" in corr_matrix.columns) else pd.Series(dtype=float)

# KPIs (10)
st.subheader("KPIs — Correlation & drivers")

# Top 5 positive corr with TARGET
top5_pos = target_corr.sort_values(ascending=False).head(5)
# Top 5 negative corr with TARGET
top5_neg = target_corr.sort_values().head(5)

# Most correlated with Income & Credit
most_corr_income = corr_matrix["AMT_INCOME_TOTAL"].dropna().abs().sort_values(ascending=False).index[1] \
    if ("AMT_INCOME_TOTAL" in corr_matrix.columns and len(corr_matrix["AMT_INCOME_TOTAL"])>1) else "N/A"
most_corr_credit = corr_matrix["AMT_CREDIT"].dropna().abs().sort_values(ascending=False).index[1] \
    if ("AMT_CREDIT" in corr_matrix.columns and len(corr_matrix["AMT_CREDIT"])>1) else "N/A"

# Select a few seeded correlations (safe guard if column missing)
corr_income_credit = corr_matrix.loc["AMT_INCOME_TOTAL", "AMT_CREDIT"] if ("AMT_INCOME_TOTAL" in corr_matrix.index and "AMT_CREDIT" in corr_matrix.columns) else np.nan
corr_age_target = corr_matrix.loc["AGE_YEARS", "TARGET"] if ("AGE_YEARS" in corr_matrix.index and "TARGET" in corr_matrix.columns) else np.nan
corr_emp_target = corr_matrix.loc["EMPLOYMENT_YEARS", "TARGET"] if ("EMPLOYMENT_YEARS" in corr_matrix.index and "TARGET" in corr_matrix.columns) else np.nan
corr_family_target = corr_matrix.loc["CNT_FAM_MEMBERS", "TARGET"] if ("CNT_FAM_MEMBERS" in corr_matrix.index and "TARGET" in corr_matrix.columns) else np.nan

# Variance explained proxy: sum(r^2) of top 5 absolute correlated features
top5_abs = target_corr.abs().sort_values(ascending=False).head(5)
variance_proxy = (top5_abs**2).sum()

# Count features with |corr| > 0.5
n_high_corr = (target_corr.abs() > 0.5).sum()

# Display KPIs in a grid (two rows)
kcols = st.columns(2)
kpis = [
    ("Top POS(+) Corr with TARGET", ", ".join([f"{idx} ({val:.2f})" for idx, val in top5_pos.items()])),
    ("Top NEG(−) Corr with TARGET", ", ".join([f"{idx} ({val:.2f})" for idx, val in top5_neg.items()])),
    ("Most correlated with Income", str(most_corr_income)),
    ("Most correlated with Credit", str(most_corr_credit)),
    ("Corr(Income, Credit)", f"{corr_income_credit:.2f}" if not np.isnan(corr_income_credit) else "N/A"),
    ("Corr(Age, TARGET)", f"{corr_age_target:.2f}" if not np.isnan(corr_age_target) else "N/A"),
    ("Corr(EmploymentYears, TARGET)", f"{corr_emp_target:.2f}" if not np.isnan(corr_emp_target) else "N/A"),
    ("Corr(FamilySize, TARGET)", f"{corr_family_target:.2f}" if not np.isnan(corr_family_target) else "N/A"),
    ("Variance proxy (top5)", f"{variance_proxy:.3f}")
    # ("# features |corr|>0.5", f"{int(n_high_corr)}"),
]

for i, (lab, val) in enumerate(kpis):
    with kcols[i % 2]:
        # numbers used as metrics, lists use markdown for readability
        if isinstance(val, str) and ("," in val or " " in val and len(val) > 30):
            st.markdown(f"**{lab}**  \n<span style='font-size:14px'>{val}</span>", unsafe_allow_html=True)
        else:
            st.metric(lab, val)

st.markdown("---")

# CHARTS (10)
st.subheader("Charts — correlations & drivers")

c1,c2 = st.columns(2)

with c1:
    # 1) Heatmap — correlation for selected numerics
    st.markdown("**1) Correlation heatmap (selected numeric features)**")
    heat_cols = [c for c in ["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","DTI","LTI","AGE_YEARS","EMPLOYMENT_YEARS","CNT_FAM_MEMBERS","CNT_CHILDREN","TARGET"] if c in df.columns]
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(df[heat_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar=False)
    st.pyplot(fig); plt.close(fig)

with c2:
    # 2) Bar — |Correlation| of features vs TARGET (top N)
    st.markdown("**2) |Correlation| vs TARGET (top features)**")
    abs_corr = target_corr.abs().sort_values(ascending=False)
    topn = min(20, len(abs_corr))
    fig, ax = plt.subplots(figsize=(6,3))
    abs_corr.head(topn).plot(kind="bar", color=PALETTE["navy"], ax=ax)
    ax.set_ylabel("|corr| with TARGET")
    st.pyplot(fig); plt.close(fig)

c1,c2 = st.columns(2)
with c1:
    # 3) Scatter — Age vs Credit (hue=TARGET)
    st.markdown("**3) Age vs Credit (color = TARGET)**")
    if {"AGE_YEARS","AMT_CREDIT","TARGET"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(5,3))
        colors = df["TARGET"].map({0: PALETTE["navy"], 1: PALETTE["warm_orange"]})
        ax.scatter(df["AGE_YEARS"], df["AMT_CREDIT"], c=colors, alpha=0.25)
        ax.set_xlabel("Age (years)"); ax.set_ylabel("Credit amount")
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("AGE_YEARS or AMT_CREDIT or TARGET missing for scatter.")

with c2:
    # 4) Scatter — Age vs Income (hue=TARGET)
    st.markdown("**4) Age vs Income (color = TARGET)**")
    if {"AGE_YEARS","AMT_INCOME_TOTAL","TARGET"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(5,3))
        colors = df["TARGET"].map({0: PALETTE["navy"], 1: PALETTE["warm_orange"]})
        ax.scatter(df["AGE_YEARS"], df["AMT_INCOME_TOTAL"], c=colors, alpha=0.25)
        ax.set_xlabel("Age (years)"); ax.set_ylabel("Income")
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("AGE_YEARS or AMT_INCOME_TOTAL or TARGET missing for scatter.")

c1,c2 = st.columns(2)
with c1:
    # 5) Scatter / jitter — Employment Years vs TARGET
    st.markdown("**5) Employment years vs TARGET**")
    if "EMPLOYMENT_YEARS" in df.columns and "TARGET" in df.columns:
        fig, ax = plt.subplots(figsize=(6,3))
        # jitter target on y slightly for visualization: plot employment years vs target with small jitter
        jitter = (np.random.rand(len(df)) - 0.5) * 0.08
        ax.scatter(df["EMPLOYMENT_YEARS"], df["TARGET"] + jitter, alpha=0.15, c=df["TARGET"].map({0:PALETTE["navy"],1:PALETTE["warm_orange"]}))
        ax.set_xlabel("Employment years"); ax.set_ylabel("TARGET (jittered)")
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("EMPLOYMENT_YEARS or TARGET missing.")

with c2:
    # 6) Boxplot — Credit by Education
    st.markdown("**6) Credit by Education**")
    if "NAME_EDUCATION_TYPE" in df.columns and "AMT_CREDIT" in df.columns:
        fig, ax = plt.subplots(figsize=(6,3))
        order = df.groupby("NAME_EDUCATION_TYPE")["AMT_CREDIT"].median().sort_values(ascending=False).index
        sns.boxplot(x="NAME_EDUCATION_TYPE", y="AMT_CREDIT", data=df, order=order, palette=[PALETTE["navy"]], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("NAME_EDUCATION_TYPE or AMT_CREDIT missing.")

c1,c2 = st.columns(2)
with c1:
    # 7) Boxplot — Income by Family Status
    st.markdown("**7) Income by Family Status**")
    if "NAME_FAMILY_STATUS" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        fig, ax = plt.subplots(figsize=(6,3))
        order = df.groupby("NAME_FAMILY_STATUS")["AMT_INCOME_TOTAL"].median().sort_values(ascending=False).index
        sns.boxplot(x="NAME_FAMILY_STATUS", y="AMT_INCOME_TOTAL", data=df, order=order, palette=[PALETTE["warm_orange"]], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("NAME_FAMILY_STATUS or AMT_INCOME_TOTAL missing.")

with c2:
    # 8) Pair Plot — Income, Credit, Annuity, TARGET (sample for speed)
    st.markdown("**8) Pair plot**")
    pair_vars = [c for c in ["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","TARGET"] if c in df.columns]
    if len(pair_vars) >= 2:
        sample = df[pair_vars].dropna().sample(n=min(1000, len(df)), random_state=42)
        # seaborn pairplot can be heavy: sample before plotting
        pp_fig = sns.pairplot(sample, hue="TARGET", corner=True, plot_kws={"alpha":0.6, "s":20})
        st.pyplot(pp_fig.fig)
        plt.close(pp_fig.fig)
    else:
        st.info("Not enough vars for pairplot.")

c1,c2 = st.columns(2)
with c1:
    # 9) Filtered Bar — Default Rate by Gender (responsive to sidebar)
    st.markdown("**9) Default rate by Gender**")
    if "CODE_GENDER" in df.columns:
        grp = df.groupby("CODE_GENDER")["TARGET"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(4,3))
        grp.plot(kind="bar", color=[PALETTE["navy"], PALETTE["warm_orange"]], ax=ax)
        ax.set_ylabel("Default rate")
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("CODE_GENDER missing.")

with c2:
    # 10) Filtered Bar — Default Rate by Education (responsive)
    st.markdown("**10) Default rate by Education**")
    if "NAME_EDUCATION_TYPE" in df.columns:
        grp = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(6,3))
        grp.plot(kind="bar", color=PALETTE["gold"], ax=ax)
        ax.set_ylabel("Default rate")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig); plt.close(fig)
    else:
        st.info("NAME_EDUCATION_TYPE missing.")

st.markdown("---")

# Narrative: candidate policy rules
st.subheader("Narrative — candidate policy rules")
st.markdown("""
- Features with strong positive correlation to TARGET (defaults) are candidates for **hard/soft rules**:
  - e.g., impose **LTI caps** (maximum loan-to-income) or **minimum income floors** for high-risk segments.
- Use **income & credit correlation** to set product pricing tiers — higher LTI requests may require additional checks or co-applicants.
- Consider **employment tenure** and **age** as underwriting filters or for tiered interest rates.
- For features with very high correlation, perform more advanced modeling (logistic regression / SHAP) before automating policies.
""")
