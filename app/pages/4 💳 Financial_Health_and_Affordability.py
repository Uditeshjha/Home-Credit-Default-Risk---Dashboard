import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from utils import preprocessing as pp, load_data as ld
from utils.theme import apply_theme,get_palette

# Apply theme
apply_theme()
PALETTE = get_palette()

st.title("Financial Health & Affordability")

# Load processed dataset
df = ld.load_data("data/processed/full_cleaned.csv")

# Apply global filters
filters = pp.get_global_filters(df)
df = pp.apply_global_filters(df, filters)

# KPIs
st.subheader("Key Financial KPIs")
col1, col2, col3, col4, col5 = st.columns(5)

avg_income = df["AMT_INCOME_TOTAL"].mean()
median_income = df["AMT_INCOME_TOTAL"].median()
avg_credit = df["AMT_CREDIT"].mean()
avg_annuity = df["AMT_ANNUITY"].mean()
avg_goods = df["AMT_GOODS_PRICE"].mean()

avg_dti = (df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]).mean()
avg_lti = (df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]).mean()

# Gaps
inc_gap = (
    df.loc[df["TARGET"] == 0, "AMT_INCOME_TOTAL"].mean()
    - df.loc[df["TARGET"] == 1, "AMT_INCOME_TOTAL"].mean()
)
cred_gap = (
    df.loc[df["TARGET"] == 0, "AMT_CREDIT"].mean()
    - df.loc[df["TARGET"] == 1, "AMT_CREDIT"].mean()
)

pct_high_credit = (df["AMT_CREDIT"] > 1_000_000).mean() * 100

with col1: st.metric("Avg Annual Income", f"{avg_income:,.0f}")
with col2: st.metric("Median Annual Income", f"{median_income:,.0f}")
with col3: st.metric("Avg Credit Amount", f"{avg_credit:,.0f}")
with col4: st.metric("Avg Annuity", f"{avg_annuity:,.0f}")
with col5: st.metric("Avg Goods Price", f"{avg_goods:,.0f}")

col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Avg DTI", f"{avg_dti:.2f}")
with col2: st.metric("Avg LTI", f"{avg_lti:.2f}")
with col3: st.metric("Income Gap (Non-def − Def)", f"{inc_gap:,.0f}")
with col4: st.metric("Credit Gap (Non-def − Def)", f"{cred_gap:,.0f}")
with col5: st.metric("% High Credit (>1M)", f"{pct_high_credit:.1f}%")


# Charts
st.subheader("Charts")

# 1. Income distribution
c1, c2 = st.columns(2)
with c1:
    st.markdown("**1) Income distribution**")
    fig, ax = plt.subplots(figsize=(4,3))
    ax.hist(df["AMT_INCOME_TOTAL"].dropna(), bins=30,
            color=PALETTE["navy"], edgecolor="k", alpha=0.7)
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

# 2. Credit distribution
with c2:
    st.markdown("**2) Credit distribution**")
    fig, ax = plt.subplots(figsize=(4,3))
    ax.hist(df["AMT_CREDIT"].dropna(), bins=30,
            color=PALETTE["warm_orange"], edgecolor="k", alpha=0.7)
    ax.set_xlabel("Credit Amount")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

# 3. Annuity distribution
c1, c2 = st.columns(2)
with c1:
    st.markdown("**3) Annuity distribution**")
    fig, ax = plt.subplots(figsize=(4,3))
    ax.hist(df["AMT_ANNUITY"].dropna(), bins=30,
            color=PALETTE["gold"], edgecolor="k", alpha=0.7)
    ax.set_xlabel("Annuity Amount")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

# 4. Income vs Credit scatter
with c2:
    st.markdown("**4) Income vs Credit**")
    fig, ax = plt.subplots(figsize=(4,3))
    ax.scatter(df["AMT_INCOME_TOTAL"], df["AMT_CREDIT"],
               alpha=0.3, color=PALETTE["navy"])
    ax.set_xlabel("Income")
    ax.set_ylabel("Credit")
    st.pyplot(fig)
    plt.close(fig)

# 5. Income vs Annuity scatter
c1, c2 = st.columns(2)
with c1:
    st.markdown("**5) Income vs Annuity**")
    fig, ax = plt.subplots(figsize=(4,3))
    ax.scatter(df["AMT_INCOME_TOTAL"], df["AMT_ANNUITY"],
               alpha=0.3, color=PALETTE["warm_orange"])
    ax.set_xlabel("Income")
    ax.set_ylabel("Annuity")
    st.pyplot(fig)
    plt.close(fig)

# 6. Credit by Target (boxplot)
with c2:
    st.markdown("**6) Credit by Target**")
    fig, ax = plt.subplots(figsize=(4,3))
    sns.boxplot(x="TARGET", y="AMT_CREDIT", data=df,
                palette=[PALETTE["navy"], PALETTE["warm_orange"]], ax=ax)
    ax.set_xticklabels(["Repaid", "Default"])
    st.pyplot(fig)
    plt.close(fig)

# 7. Income by Target (boxplot)
c1, c2 = st.columns(2)
with c1:
    st.markdown("**7) Income by Target**")
    fig, ax = plt.subplots(figsize=(4,3))
    sns.boxplot(x="TARGET", y="AMT_INCOME_TOTAL", data=df,
                palette=[PALETTE["navy"], PALETTE["warm_orange"]], ax=ax)
    ax.set_xticklabels(["Repaid", "Default"])
    st.pyplot(fig)
    plt.close(fig)

# 8. Joint density Income–Credit (KDE)
with c2:
    st.markdown("**8) Joint Density: Income vs Credit (hexbin, log scale)**")
    if {"AMT_INCOME_TOTAL", "AMT_CREDIT"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(4, 3))

        # log1p transforms reduce skew and make patterns visible
        x = np.log1p(df["AMT_INCOME_TOTAL"].clip(lower=0))
        y = np.log1p(df["AMT_CREDIT"].clip(lower=0))

        # hexbin: gridsize controls resolution; mincnt avoids plotting empty bins
        hb = ax.hexbin(x, y, gridsize=50, bins='log', cmap='Blues', mincnt=1)

        # colorbar shows counts (log scale when bins='log')
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('log(count)')

        ax.set_xlabel("log(Income + 1)")
        ax.set_ylabel("log(Credit + 1)")
        ax.set_title("Income vs Credit (hexbin)")

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("AMT_INCOME_TOTAL or AMT_CREDIT missing.")

# 9. Income Brackets vs Default Rate
c1, c2 = st.columns(2)
with c1:
    st.markdown("**9) Income Brackets vs Default Rate**")
    inc_br = df.groupby("INCOME_BRACKET")["TARGET"].mean()
    fig, ax = plt.subplots(figsize=(4,3))
    inc_br.plot(kind="bar", color=PALETTE["gold"], ax=ax)
    ax.set_ylabel("Default Rate")
    st.pyplot(fig)
    plt.close(fig)

# 10. Correlation heatmap
with c2:
    st.markdown("**10) Financial variable correlations**")
    cols = ["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY",
            "DTI","LTI","TARGET"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, cbar=False)
    st.pyplot(fig)
    plt.close(fig)

# Narrative
st.subheader("Key Insights")
st.markdown("""
- Applicants with **higher Credit amounts relative to Income (high LTI)** 
  or **higher Annuity relative to Income (high DTI)** are at greater risk of default.
- The **income gap** shows that defaulters tend to have significantly lower average income.
- High-credit applicants (>1M loan) form a small but risky segment.
- Overall, affordability metrics (DTI, LTI) are strong indicators of repayment capacity.
""")