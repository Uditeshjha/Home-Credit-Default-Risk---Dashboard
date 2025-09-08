# pages/2_Target_and_Risk_Segmentation.py
"""
Page 2 — Target & Risk Segmentation
Focus: Show how default (TARGET=1) is associated with risks across various key segments.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# utils
from utils import preprocessing as pp
from utils import load_data as ld
from utils.theme import apply_theme, get_palette

# Apply theme
apply_theme()
PALETTE = get_palette()

# Page header
st.title("Target & Risk Segmentation")
st.write("Explore how repayment vs default varies across demographics and financial features.")

# Load processed dataset
df_full = ld.load_data("../data/processed/full_cleaned.csv")

# Global filters
filters = pp.get_global_filters(df_full)
df = pp.apply_global_filters(df_full, filters)

if df.shape[0] == 0:
    st.warning("No rows match current filters.")
    st.stop()

# KPIs
st.subheader("Key Performance Indicators (KPIs)")

def default_rate(subset):
    return (subset["TARGET"] == 1).mean() if len(subset) > 0 else float("nan")

total = len(df)
default_r = default_rate(df)
repay_r = 1 - default_r

male_r = default_rate(df[df["CODE_GENDER"] == "M"])
female_r = default_rate(df[df["CODE_GENDER"] == "F"])
low_r = default_rate(df[df["INCOME_BRACKET"] == "Low"])
mid_r = default_rate(df[df["INCOME_BRACKET"] == "Mid"])
high_r = default_rate(df[df["INCOME_BRACKET"] == "High"])

# find riskiest education and housing
edu_risks = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().sort_values(ascending=False)
house_risks = df.groupby("NAME_HOUSING_TYPE")["TARGET"].mean().sort_values(ascending=False)
top_edu = edu_risks.index[0] if not edu_risks.empty else "N/A"
top_edu_rate = edu_risks.iloc[0]*100 if not edu_risks.empty else 0
top_house = house_risks.index[0] if not house_risks.empty else "N/A"
top_house_rate = house_risks.iloc[0]*100 if not house_risks.empty else 0

cols = st.columns(2)
kpis = [
    ("Default rate", f"{default_r*100:.2f}%"),
    ("Repayment rate", f"{repay_r*100:.2f}%"),
    ("Default rate (Male)", f"{male_r*100:.2f}%" if pd.notna(male_r) else "N/A"),
    ("Default rate (Female)", f"{female_r*100:.2f}%" if pd.notna(female_r) else "N/A"),
    ("Low income default rate", f"{low_r*100:.2f}%" if pd.notna(low_r) else "N/A"),
    ("Mid income default rate", f"{mid_r*100:.2f}%" if pd.notna(mid_r) else "N/A"),
    ("High income default rate", f"{high_r*100:.2f}%" if pd.notna(high_r) else "N/A"),
    ("Riskiest education", f"{top_edu} ({top_edu_rate:.1f}%)"),
    ("Riskiest housing", f"{top_house} ({top_house_rate:.1f}%)"),
    ("Total defaults", f"{(df['TARGET']==1).sum():,}"),
]

for i, (label, value) in enumerate(kpis):
    with cols[i % 2]:
        if label in ["Riskiest education", "Riskiest housing"]:
            # unsafe_allow_html=True → allows HTML in Markdown.
            st.markdown(f"**{label}:** <br><span style='font-size:25px; color:#fff;'>{value}</span>",
                        unsafe_allow_html=True)
        else:
            st.metric(label, value)

st.markdown("---")

# Charts (10)
st.subheader("Target & Risk Segmentation Charts")

c1,c2 = st.columns(2)
with c1:
    st.markdown("**1) Target distribution**")
    # Chart 1: Target distribution (donut)
    colors = [PALETTE["navy"], PALETTE["warm_orange"]]
    vc = df["TARGET"].value_counts().sort_index()
    label_map = {0: "Repaid", 1: "Default"}
    labels = [label_map.get(i, str(i)) for i in vc.index]

    fig, ax = plt.subplots(figsize=(3, 3))
    wedges, texts, autotexts = ax.pie(
        vc.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors[:len(vc.values)]
    )
    for t in texts: t.set(color="black", fontsize=9, weight="bold")
    for at in autotexts: at.set(color="white", fontsize=9, weight="bold")
    ax.set(aspect="equal", title="Target distribution")
    st.pyplot(fig)
    plt.close(fig)

with c2:
    st.markdown("**2) Default rate by gender**")
    # Chart 2: Default rate by gender
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(x="CODE_GENDER", y="TARGET", data=df,
                color=PALETTE["warm_orange"], ax=ax)
    ax.set_title("Default rate by Gender"); ax.set_ylabel("Default rate")
    st.pyplot(fig)
    plt.close(fig)

c3,c4 = st.columns(2)

with c3:
    st.markdown("**3) Default rate by income bracket**")
    # Chart 3: Default rate by income bracket
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(x="INCOME_BRACKET", y="TARGET", data=df,
                color=PALETTE["gold"], ax=ax)
    ax.set_title("Default rate by Income Bracket"); ax.set_ylabel("Default rate")
    st.pyplot(fig); plt.close(fig)

with c4:
    st.markdown("**4) Default rate by education**")
    # Chart 4: Default rate by education
    fig, ax = plt.subplots(figsize=(4, 3))
    order = df["NAME_EDUCATION_TYPE"].value_counts().index
    sns.barplot(x="NAME_EDUCATION_TYPE", y="TARGET", data=df, order=order,
                 color=PALETTE["navy"], ax=ax)
    ax.set_title("Default rate by Education") 
    ax.set_ylabel("Default rate") 
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    st.pyplot(fig); plt.close(fig)

c5,c6 = st.columns(2)
with c5:
    st.markdown("**5) Default rate by family status**")
    # Chart 5: Default rate by family status
    fig, ax = plt.subplots(figsize=(4, 3))
    order = df["NAME_FAMILY_STATUS"].value_counts().index
    sns.barplot(x="NAME_FAMILY_STATUS", y="TARGET", data=df, order=order,
                 color=PALETTE["warm_orange"], ax=ax)
    ax.set_title("Default rate by Family Status"); ax.set_ylabel("Default rate") 
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    st.pyplot(fig); plt.close(fig)

with c6:
    st.markdown("**6) Default rate by housing type**")
    # Chart 6: Default rate by housing type
    fig, ax = plt.subplots(figsize=(4, 3))
    order = df["NAME_HOUSING_TYPE"].value_counts().index
    sns.barplot(x="NAME_HOUSING_TYPE", y="TARGET", data=df, order=order,
                 color=PALETTE["gold"], ax=ax)
    ax.set_title("Default rate by Housing Type"); ax.set_ylabel("Default rate") 
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    st.pyplot(fig); plt.close(fig)

c7,c8 = st.columns(2)
with c7:
    st.markdown("**7) Age distribution by target**")
    # Chart 7: Age distribution segmented by target
    if "AGE_YEARS" in df.columns:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.histplot(data=df, x="AGE_YEARS", hue="TARGET", multiple="stack",
                    bins=30, palette=[PALETTE["navy"], PALETTE["warm_orange"]], ax=ax)
        ax.set_title("Age distribution by Target")
        st.pyplot(fig); plt.close(fig)

with c8:
    st.markdown("**8) Income distribution by target**")
    # Chart 8: Boxplot of income vs target
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.boxplot(x="TARGET", y="AMT_INCOME_TOTAL", data=df,
                palette=[PALETTE["navy"], PALETTE["warm_orange"]], ax=ax)
    ax.set_title("Income distribution by Target")
    ax.set_xticklabels(["Repaid", "Default"])
    st.pyplot(fig); plt.close(fig)

c9,c10 = st.columns(2)
with c9:
    st.markdown("**9) Credit distribution by target**")
    # Chart 9: Boxplot of credit vs target
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.boxplot(x="TARGET", y="AMT_CREDIT", data=df,
                palette=[PALETTE["navy"], PALETTE["warm_orange"]], ax=ax)
    ax.set_title("Credit distribution by Target")
    ax.set_xticklabels(["Repaid", "Default"])
    st.pyplot(fig); plt.close(fig)

with c10:
    st.markdown("**10) Employment years vs default rate**")
    # Chart 10: Employment years vs default rate (binned)
    if "EMPLOYMENT_YEARS" in df.columns:
        df["EMP_BIN"] = pd.cut(df["EMPLOYMENT_YEARS"], bins=[0,5,10,20,30,50], labels=["0-5","5-10","10-20","20-30","30+"])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(x="EMP_BIN", y="TARGET", data=df,
                     color=PALETTE["navy"], ax=ax)
        ax.set_title("Default rate by Employment Tenure"); ax.set_ylabel("Default rate")
        st.pyplot(fig); plt.close(fig)

st.markdown("---")

# Insights
st.subheader("Key Insights")
st.markdown(f"- Overall default rate: **{default_r*100:.2f}%** (repaid: {repay_r*100:.2f}%).")
st.markdown(f"- Female default rate: **{female_r*100:.2f}%** vs Male: **{male_r*100:.2f}%**.")
st.markdown(f"- Riskiest education segment: **{top_edu}** ({top_edu_rate:.1f}% default).")
st.markdown(f"- Riskiest housing segment: **{top_house}** ({top_house_rate:.1f}% default).")
