# Home Credit Default Risk — Interactive Streamlit Dashboard

**Compact, professional multi-page Streamlit dashboard to explore Home Credit loan applicants, analyze default risk, and investigate affordability & correlations.**  
Built with Python, Pandas, Streamlit and Matplotlib/Seaborn.

---

## 📚 Pages (menu) & short descriptions
- **🏦 Home** — Project intro, high-level KPIs, dataset & processing info.  
- **🧾 Overview & Data Quality** — Missingness, data types, initial distributions, quick quality checks.  
- **🎯 Target & Risk Segmentation** — Target distribution, default rates across segments, riskiest cohorts.  
- **👪 Demographics & Household Profile** — Age, gender, family, children, occupation, household-level patterns.  
- **💳 Financial Health & Affordability** — Income / credit / annuity distributions, DTI / LTI analyses, income/credit gaps.  
- **🔗 Correlations, Drivers & Slice-and-Dice** — Correlation heatmaps, top drivers, and interactive filters.

> Each page follows the assignment requirement: **10 KPIs + 10 visualizations per page**, plus short business narratives.

---

## 📌 What's implemented (high level)
- Modular multi-page Streamlit app (one page per business topic).
- Compressed the raw dataset and optimized the memory.  
- Professional, corporate warm color theme and consistent visuals.  
- Preprocessing pipeline (`utils/preprocessing.py`) that:
  - Creates derived columns: `AGE_YEARS`, `EMPLOYMENT_YEARS`, `DTI`, `LTI`, `ANNUITY_TO_CREDIT`.  
  - Handles missing values (drop very-high-missing columns, impute median/mode for others).  
  - Standardizes categories (merge rare labels to “Other”).
  - Defines `INCOME_BRACKET` quantiles (Low / Mid / High).  
- Global sidebar filters (persist across pages): Gender, Education, Family Status, Housing Type, Age Range, Income Bracket, Employment Tenure.  
- Faster visual alternatives for heavy plots (hexbin/hist2d for joint densities, sampling where needed).  
- Processed outputs saved as `full_cleaned.csv` and `full_cleaned.pkl` (via `pp.save_processed_data()`).  
- Simple caching / load-from-processed logic so app is responsive after first preprocess.

---

## 🔍 Derived columns (how they are computed)
- `AGE_YEARS = -DAYS_BIRTH / 365.25`  
- `EMPLOYMENT_YEARS = clip(-DAYS_EMPLOYED / 365.25)` (clip sentinel values used for “not employed”)  
- `DTI = AMT_ANNUITY / AMT_INCOME_TOTAL` (debt-to-income)  
- `LTI = AMT_CREDIT / AMT_INCOME_TOTAL` (loan-to-income)  
- `ANNUITY_TO_CREDIT = AMT_ANNUITY / AMT_CREDIT`

These are created in preprocessing and used across pages.

---

## 🔧 Tech stack & dependencies
- Python 3.13+ (recommended)  
- Core libs: `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`
- See `requirements.txt` for exact, pinned packages.

---

## 📥 Dataset
**Name:** `application_train.csv` (Home Credit Default Risk)  
**Suggest download:**  
https://drive.google.com/file/d/1jegRRGWBNYQy4Y-_ngyL0wPZaauhb6xu/view


---

## ▶️ How to run locally

1. Clone the repo:
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2. Create & activate a virtual environment:
### macOS / Linux
python -m venv .venv
source .venv/bin/activate

### Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

3. Install dependencies:
pip install -r requirements.txt

4. Ensure dataset is present at data/raw/application_train.csv

5. Run the app:
streamlit run app/Home.py

## 🚀 Future scope / enhancements
- Add interactive Plotly charts (zoom, hover tooltips) for drill-down.

- Add export buttons (download filtered CSV, export charts to PNG).

- Add authentication/role-based access for sensitive datasets.

- Automate preprocessing via CI or background job; add versioning for processed artifacts.

- Add unit tests for preprocessing and integration tests for pages.

- Add Dockerfile for reproducible deployment.

- Integrate modeling pipeline (train/test + SHAP explanations) and provide model scoring simulator.

- Improve performance with precomputed groupbys, Downsampling of dataset.

## ⚡Live Website URL (hosted in streamlit cloud)

#### https://home-credit-default-risk---dashboard.streamlit.app/

📬 Contact & credits

**Built by: Uditesh Jha**

**Dataset: Home Credit Default Risk(application_train.csv)**
