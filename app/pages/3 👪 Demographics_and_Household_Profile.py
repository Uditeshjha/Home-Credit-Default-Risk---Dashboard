# pages/3_Demographics_and_Household.py
"""
Page 3 — Demographics & Household Profile
Focus: Information about applicants, household structure and human factors.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import preprocessing as pp, load_data as ld
from utils.theme import apply_theme, get_palette

# Apply theme
apply_theme()
PALETTE = get_palette()

st.title("Demographics & Household Profile")

# Load processed dataset
df = ld.load_data("data/processed/full_cleaned.csv")

# Apply global filters
filters = pp.get_global_filters(df)
df = pp.apply_global_filters(df, filters)

# KPIs
st.subheader("Key Demographics KPIs")
col1, col2, col3, col4, col5 = st.columns(5)

#def_rate = lambda subset: (subset["TARGET"]==1).mean() if len(subset)>0 else float("nan")

# 1. Gender split
pct_male = (df["CODE_GENDER"]=="M").mean()*100 if "CODE_GENDER" in df else 0
pct_female = (df["CODE_GENDER"]=="F").mean()*100 if "CODE_GENDER" in df else 0

# 2-3. Avg Age by Target
avg_age_def = df.loc[df["TARGET"]==1,"AGE_YEARS"].mean()
# Target = 0 -> repaid
avg_age_nondef = df.loc[df["TARGET"]==0,"AGE_YEARS"].mean()

# 4. % with children
pct_children = (df["CNT_CHILDREN"]>0).mean()*100

# 5. Avg family size
avg_family = df["CNT_FAM_MEMBERS"].mean()

# 6. Married vs Single
married = df["NAME_FAMILY_STATUS"].str.contains("Married", case=False, na=False).mean()*100
single = df["NAME_FAMILY_STATUS"].str.contains("Single", case=False, na=False).mean()*100

# 7. Higher education
higher = df["NAME_EDUCATION_TYPE"].str.contains("Bachelor|Higher|Academic degree", case=False, na=False).mean()*100

# 8. Living with parents
with_parents = (df["NAME_HOUSING_TYPE"]=="With parents").mean()*100

# 9. Currently working (if DAYS_EMPLOYED negative → working)
working = (df["DAYS_EMPLOYED"]<0).mean()*100 if "DAYS_EMPLOYED" in df else 0

# 10. Avg employment years
avg_emp = df["EMPLOYMENT_YEARS"].mean() if "EMPLOYMENT_YEARS" in df else 0

with col1: st.metric("% Male", f"{pct_male:.1f}%")
with col2: st.metric("% Female", f"{pct_female:.1f}%")
with col3: st.metric("Avg Age (Defaulters)", f"{avg_age_def:.1f}")
with col4: st.metric("Avg Age (Non-Defaulters)", f"{avg_age_nondef:.1f}")
with col5: st.metric("% With Children", f"{pct_children:.1f}%")

col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Avg Family Size", f"{avg_family:.1f}")
with col2: st.metric("% Married", f"{married:.1f}%")
with col3: st.metric("% Single", f"{single:.1f}%")
with col4: st.metric("% Higher Education", f"{higher:.1f}%")
with col5: st.metric("% Living With Parents", f"{with_parents:.1f}%")

col1, col2 = st.columns(2)
with col1: st.metric("% Currently Working", f"{working:.1f}%")
with col2: st.metric("Avg Employment Years", f"{avg_emp:.1f}")

# Charts
st.subheader("Charts")

c1, c2 = st.columns(2)

# 1. Age distribution
with c1:
    st.markdown("**1) Age distribution**")
    fig, ax = plt.subplots(figsize=(4,3))
    ax.hist(df["AGE_YEARS"].dropna(), bins=20, color=PALETTE["navy"], edgecolor="k", alpha=0.7)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

# 2. Age vs Target (default)
with c2:
    st.markdown("**2) Age distribution by Target (overlay)**")
    fig, ax = plt.subplots(figsize=(4,3))
    df[df["TARGET"]==0]["AGE_YEARS"].hist(bins=20, alpha=0.7, color=PALETTE["navy"], label="Repaid", ax=ax)
    df[df["TARGET"]==1]["AGE_YEARS"].hist(bins=20, alpha=0.7, color=PALETTE["warm_orange"], label="Default", ax=ax)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
# with c2:
#     st.markdown("**2) Age vs Default (boxplot)**")
#     fig, ax = plt.subplots(figsize=(4,3))
#     sns.boxplot(x="TARGET", y="AGE_YEARS", data=df, palette=[PALETTE["navy"], PALETTE["warm_orange"]], ax=ax)
#     ax.set_xticklabels(["Repaid", "Default"])
#     ax.set_xlabel("")
#     ax.set_ylabel("Age (years)")
#     st.pyplot(fig)
#     plt.close(fig)

# 3. Education distribution
c1, c2 = st.columns(2)
# with c1:
#     st.markdown("**3) Education distribution**")
#     vc = df["NAME_EDUCATION_TYPE"].value_counts().head(10)
#     fig, ax = plt.subplots(figsize=(4,3))
#     vc.plot(kind="bar", color=PALETTE["gold"], ax=ax)
#     ax.set_ylabel("Count")
#     st.pyplot(fig)
#     plt.close(fig)
with c1:
    st.markdown("**3) Gender distribution**")
    vc = df["CODE_GENDER"].value_counts()
    fig, ax = plt.subplots(figsize=(4,3))
    vc.plot(kind="bar", color=[PALETTE["navy"], PALETTE["warm_orange"]], ax=ax)
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

# 4. Default rate by Education
with c2:
    st.markdown("**4) Default rate by Education**")
    edu = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(4,3))
    edu.plot(kind="bar", color=PALETTE["warm_orange"], ax=ax)
    ax.set_ylabel("Default Rate")
    st.pyplot(fig)
    plt.close(fig)

# 5. Family Status distribution
c1, c2 = st.columns(2)
with c1:
    st.markdown("**5) Family Status distribution**")
    vc = df.groupby("NAME_FAMILY_STATUS")["TARGET"].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(4,3))
    vc.plot(kind="bar", color=PALETTE["navy"], ax=ax)
    ax.set_ylabel("Defaut Rate")
    st.pyplot(fig)
    plt.close(fig)

# 6. Default rate by Family Status
with c2:
    # st.markdown("**6) Default rate by Family Status**")
    # fam = df.groupby("NAME_FAMILY_STATUS")["TARGET"].mean().sort_values(ascending=False)
    # fig, ax = plt.subplots(figsize=(4,3))
    # fam.plot(kind="bar", color=PALETTE["warm_orange"], ax=ax)
    # ax.set_ylabel("Default Rate")
    # st.pyplot(fig)
    # plt.close(fig)
    st.markdown("**6) Occupation distribution (Top 10)**")
    if "OCCUPATION_TYPE" in df.columns:
        vc = df["OCCUPATION_TYPE"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(4,3))
        vc.plot(kind="bar", color=PALETTE["gold"], ax=ax)
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close(fig)

# 7. Housing distribution
c1, c2 = st.columns(2)
with c1:
    st.markdown("**7) Housing distribution**")
    vc = df["NAME_HOUSING_TYPE"].value_counts()
    fig, ax = plt.subplots(figsize=(4,3))
    vc.plot(kind="bar", color=PALETTE["gold"], ax=ax)
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)
    # st.markdown("**7) Housing type distribution**")
    # vc = df["NAME_HOUSING_TYPE"].value_counts()
    # colors = [PALETTE["navy"], PALETTE["warm_orange"]]
    # fig, ax = plt.subplots(figsize=(4,4))
    # wedges, texts, autotexts = ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%", startangle=90, 
    #                                   colors=colors[:len(vc.values)])
    # for t in texts: t.set(color="black", fontsize=9, weight="bold")
    # for at in autotexts: at.set(color="white", fontsize=9, weight="bold")
    # ax.set(aspect="equal")
    # st.pyplot(fig)
    # plt.close(fig)

# 8. Default rate by Housing
with c2:
    st.markdown("**8) Default rate by Housing**")
    house = df.groupby("NAME_HOUSING_TYPE")["TARGET"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(4,3))
    house.plot(kind="bar", color=PALETTE["warm_orange"], ax=ax)
    ax.set_ylabel("Default Rate")
    st.pyplot(fig)
    plt.close(fig)

# 9. Children distribution
c1, c2 = st.columns(2)
with c1:
    st.markdown("**9) Children distribution**")
    fig, ax = plt.subplots(figsize=(4,3))
    ax.hist(df["CNT_CHILDREN"].dropna(), bins=range(0, df["CNT_CHILDREN"].max()+2), color=PALETTE["navy"], edgecolor="k", alpha=0.7)
    ax.set_xlabel("Number of Children")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

# 10. Family size distribution
with c2:
    # st.markdown("**10) Family size distribution**")
    # fig, ax = plt.subplots(figsize=(4,3))
    # ax.hist(df["CNT_FAM_MEMBERS"].dropna(), bins=range(1, int(df["CNT_FAM_MEMBERS"].max())+2), color=PALETTE["gold"], edgecolor="k", alpha=0.7)
    # ax.set_xlabel("Family Size")
    # ax.set_ylabel("Count")
    # st.pyplot(fig)
    # plt.close(fig)
    st.markdown("**10) Correlation heatmap (Age, Children, Family Size, Target)**")
    cols = ["AGE_YEARS", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "TARGET"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, cbar=False)
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Key Insights")
st.markdown("""
- Younger applicants tend to show higher default risk compared to older ones.
- Family status influences risk: single and divorced applicants have higher default rates.
- Larger family size and presence of children correlate with repayment challenges, 
  reflecting the financial burden of dependents.
""")
