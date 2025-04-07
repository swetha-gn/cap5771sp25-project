# AI-powered Job Seek Tool – A conversational agent

---

## Directory Structure

```
.
├── scripts/
│   └── milestone1/
│       ├── LinkedIn.ipynb
│       ├── Coursera_details.ipynb
│       ├── Role_based_skills.ipynb
│       └── Chatbot_job_data.ipynb
└── milestone2/
│       ├── LinkedIn.ipynb
│       ├── Coursera_details.ipynb
│       ├── Role_based_skills.ipynb
│       └── Chatbot_job_data.ipynb
├── reports/
│   └── milestone1.pdf
    └── milestone2.pdf
└── README.md
```

---

## Datasets Overview

Dataset drive link: https://drive.google.com/drive/folders/14zRRBGEjgJt_60Pm0WBqY4P0zPQm0_sz?usp=sharing (since this is have huge datasets, we are using google drive)

| Dataset | Description |
|--------|-------------|
| **LinkedIn** | Synthetic profiles with job titles, skills, certifications, and experience. |
| **Coursera** | Course metadata including title, rating, reviews, skills, and qualifications. |
| **Role-Based Skills** | Job descriptions with required skills, responsibilities, and education levels. |
| **Chatbot Job Data** | Raw scraped job listings with text fields, salary data, geolocation, and timestamps. |

---

## Data Preprocessing

Each dataset was cleaned and transformed to ensure consistency and completeness:
- **Standardized column names**
- **Handled missing values** via mean/mode/zero imputation
- **Converted text fields to numeric** using parsing and regex (e.g., extracting years of experience)
- **Cleaned skills and qualifications** from semi-structured formats
- **Parsed salary and date columns** into machine-readable types
- **Geolocation and temporal features** extracted from raw data (Chatbot set)

---

## Feature Engineering

A wide range of engineered features were created to enrich the raw data:

| Type | Engineered Features |
|------|---------------------|
| **Experience** | `Experience_Years`, `Seniority_Keyword_Count`, `Responsibility_Length` |
| **Skills & Courses** | `Skill_Count`, `Filtered_Skills`, `Skills_Encoded` |
| **Profile/Job Enrichment** | `Profile_Richness`, `Exp_Skill_Ratio`, `Salary_Range`, `Edu_Level` |
| **Text Features** | Word counts from descriptions, binary flags (`Has_Responsibilities`, `Has_Description`) |
| **Temporal** | `Posting_Year`, `Posting_Month`, `Posting_Weekday` (Chatbot dataset) |
| **Encoding** | Label Encoding and One-Hot Encoding for all relevant categorical variables |

---

## Feature Selection

To identify the most relevant predictors, we used **four complementary methods**:
- **Random Forest Feature Importance**
- **LASSO Regression Coefficients**
- **Chi-Square Test Scores**
- **Principal Component Analysis (PCA)** for dimensionality reduction

> Across all datasets, features like `Experience_Years`, `Skill_Count`, and `Seniority_Keyword_Count` consistently ranked high in importance.

---

## Data Modeling

We applied both **classification** and **regression** models depending on the target variable:

| Task | Target | Models Used |
|------|--------|-------------|
| Classification | `Profile_Tier`, `Experience_Binary`, `Popularity_Class` | Logistic Regression, Random Forest, SVM |
| Regression | `Average_Salary` | Linear Regression, Random Forest Regressor |

### Key Metrics Reported:
- **Accuracy, Precision, Recall, F1-Score (per class)**
- **ROC AUC Scores**
- **RMSE and R² for regression models**
- **Confusion Matrices and ROC Curves for evaluation**

> In most cases, **Random Forest** models provided the best trade-off between accuracy and training time, especially under limited computational resources.

---

## Summary of Results

- **Logistic Regression and Random Forest** achieved >99% accuracy for entry-level classification tasks.
- **LASSO** confirmed sparsity in feature importance, with `Experience_Years` emerging as the strongest predictor.
- **Random Forest Regressor** performed best for salary prediction with minimal tuning.
- ROC-AUC analysis revealed stronger performance on clearly separable classes (e.g., "Low" popularity) and challenges with borderline cases (e.g., "Medium" tier).

---

## Notes for Graders

- Please explore each notebook in `scripts/milestone2/` to view complete Milestone 2 processing pipelines.
- All post-preprocessing steps (feature engineering, modeling, and evaluation) are detailed and well-annotated in each notebook.
- A compiled summary of all results, insights, and visuals is available in `reports/milestone2.pdf`.

---

## Tools & Libraries

- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- Optional: `notebook`, `jupyterlab`, `datasets` (Hugging Face)

---

##  Author

**Swetha Gendlur Nagarajan**  
University of Florida  
M.S. Applied Data Science  
April 2025
