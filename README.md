# Intelligent_and_Explainable_ML_Framework

# Explainable Telecom Churn Prediction with GNN and Uplift Modeling

## 1. Project Overview

This repository contains a complete end-to-end machine learning project for telecom customer churn prediction, explanation, and intervention planning. It combines traditional predictive modeling, explainable AI, graph-based network analysis, and uplift modeling into a unified framework.

The overall objective is to answer three business questions:

1. Which customers are likely to churn?
2. Why are they likely to churn?
3. Which customers are most likely to respond to a retention action?

This project is designed not only for prediction, but also for actionable decision-making. It equips analysts, data scientists, and business teams with interpretation, segmentation, and intervention recommendations.

---

## 2. Problem Statement and Business Value

Telecom companies face persistent customer churn because customers can switch providers easily when they feel underserved, overcharged, or disengaged. Churn is costly because it reduces recurring revenue and increases acquisition costs.

The project addresses this problem by building a pipeline that:

- predicts churn risk at the customer level
- explains the main drivers of that risk using SHAP values
- identifies customers who may be influenced by retention offers
- surfaces network-based patterns that may expose peer effects or regional influence
- presents everything through an interactive dashboard

### Business value

- improves retention targeting
- highlights customer segments with the highest churn risk
- supports proactive intervention strategy
- makes model predictions interpretable for stakeholders
- connects ML outputs to operational retention decisions

---

## 3. What the project includes

### Core workflow

- data loading and preprocessing
- exploratory data analysis
- feature engineering and synthetic sentiment features
- training and comparison of churn prediction models
- SHAP-based explanation of model decisions
- customer network analysis using graph-based methods
- uplift modeling to estimate treatment effects and intervention response
- an interactive dashboard to explore the full workflow

### Main capabilities

- predict churn probability for a customer profile
- explain model output with SHAP values
- visualize churn-related patterns across key attributes
- analyze customer similarity/network relationships
- rank customers by likely uplift from retention offers
- review findings via a Streamlit app without needing notebooks

---

## 4. Dataset Description

The project uses the public Telco Customer Churn dataset, which contains customer-level information such as account tenure, contract type, monthly charges, service usage, payment history, and churn status.

### Dataset characteristics

- Records: approximately 7,043 customers
- Original features: 20+ customer attributes
- Target variable: customer churn status
- Typical business problem: class imbalance, where churners are fewer than non-churners

### Feature categories

| Category | Examples |
|----------|----------|
| Customer demographics | gender, SeniorCitizen, Partner, Dependents |
| Account information | tenure, MonthlyCharges, TotalCharges |
| Services | PhoneService, InternetService, OnlineSecurity, etc. |
| Contract and billing | Contract, PaperlessBilling, PaymentMethod |
| Target label | Churn |

### Target distribution

The dataset is imbalanced, which is typical in churn settings. This is handled through careful splitting, evaluation, and threshold-aware interpretation.

---

## 5. Methodology

### 5.1 Exploratory Data Analysis (EDA)

The notebook and dashboard workflow begins with exploratory analysis to understand:

- churn distribution
- distribution of tenure and charges
- service adoption patterns
- contract behavior
- correlations between variables and churn risk

### 5.2 Feature Engineering

Several engineered features are created to improve predictive power and business interpretability. These include:

- tenure-based features such as early-phase flags and lifecycle bins
- charge-based features such as cost-per-tenure and average monthly value
- service adoption count and service-specific indicator features
- contract risk scores
- customer value scoring
- synthetic sentiment features derived from customer-like behavior patterns

These features are designed to reflect real-world retention considerations such as switching cost, commitment level, and satisfaction signals.

### 5.3 Model Development

The project compares several supervised learning models for binary churn classification:

| Model | Type | Purpose |
|-------|------|---------|
| Logistic Regression | interpretable linear model | baseline and explainable benchmark |
| Random Forest | ensemble tree model | captures nonlinear interactions |
| XGBoost | gradient boosting | strong predictive performance and feature importance |

### 5.4 Explainability with SHAP

SHAP is used to explain how each feature contributes to a prediction. This enables both:

- global explanations: which features matter most overall
- local explanations: why a specific customer is predicted as high-risk or low-risk

### 5.5 Graph-based customer analysis

The project also introduces network-aware analysis through a graph-based module. It builds a customer similarity network and derives graph-based signals that may capture peer influence, regional patterns, and structural similarity.

### 5.6 Uplift modeling

In addition to classification, the project includes uplift modeling to estimate treatment effects. The idea is to identify customers who are most likely to benefit from a retention intervention such as a discount, service bundle offer, or account review.

---

## 6. Repository Structure

```text
telco-churn-explainability/
├── dashboard/
│   ├── app.py
│   └── app_full.py
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│       ├── engineered_data.csv
│       └── engineered_data_with_gnn.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_explainability.ipynb
│   ├── 05_gnn_network_analysis.ipynb
│   └── 06_uplift_modeling.ipynb
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
├── src/
│   ├── feature_engineering.py
│   ├── gnn_network.py
│   ├── preprocessing.py
│   ├── sentiment.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── uplift_models.py
├── requirements.txt
└── README.md
```

---

## 7. Setup and Installation

### Prerequisites

- Python 3.10+ (3.11 recommended)
- Windows, macOS, or Linux
- 8GB RAM minimum; 16GB recommended
- Internet access for installing dependencies if needed

### Installation steps

```bash
cd telco-churn-explainability
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.\.venv\Scripts\activate

pip install -r requirements.txt
```

### NLTK setup

```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

### Optional notebook setup

```bash
pip install jupyter notebook ipykernel
```

---

## 8. Running the Project

### Option A: Run the notebooks

Start Jupyter:

```bash
jupyter notebook
```

Suggested execution order:

1. 01_eda.ipynb - exploratory analysis
2. 02_feature_engineering.ipynb - feature engineering and sentiment enrichment
3. 03_model_training.ipynb - model training and evaluation
4. 04_explainability.ipynb - SHAP analysis and interpretation
5. 05_gnn_network_analysis.ipynb - graph-based customer analysis
6. 06_uplift_modeling.ipynb - uplift estimation and offer targeting

### Option B: Launch the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard includes the following sections:

- Dataset Overview
- Exploratory Analysis
- Churn Prediction
- SHAP Explanations
- GNN Insights
- Uplift
- About

---

## 9. Model Performance and Evaluation

The saved comparison report in the repository shows strong predictive performance across the tested models.

### Representative metrics from the project outputs

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7939 | 0.6355 | 0.5267 | 0.5760 | 0.8349 |
| Random Forest | 0.7932 | 0.6426 | 0.5000 | 0.5624 | 0.8315 |
| XGBoost | 0.7882 | 0.6508 | 0.4385 | 0.5240 | 0.8360 |

These values can vary slightly depending on data splits, preprocessing, and random seeds.

### Evaluation focus

The project uses ROC-AUC as a primary metric because churn datasets are commonly imbalanced and ranking performance matters more than raw accuracy alone.

---

## 10. Key Findings and Business Insights

The project reports several recurring themes that are meaningful for retention strategy.

### Top churn drivers

1. Tenure
   - early-stage customers tend to churn more often
   - onboarding and support are critical in the first few months

2. Contract type
   - month-to-month customers are significantly more prone to churn
   - long-term contracts are associated with lower churn risk

3. Service adoption
   - customers with limited service bundles often show higher churn risk
   - additional complementary services can strengthen retention

4. Sentiment-related signals
   - negative sentiment and poor experience are important early warning indicators

5. Monthly charges and value perception
   - both very low and very high price points can be associated with higher churn

### Example retention strategies

- proactive onboarding for new customers
- targeted contract conversion offers
- service bundling and upsell campaigns
- customer satisfaction monitoring
- pricing and value communication improvements

The detailed reports are stored in the outputs directory under reports.

---

## 11. Project Modules

| Module | Purpose |
|--------|---------|
| src/preprocessing.py | data cleaning and preparation |
| src/feature_engineering.py | feature generation and encoding |
| src/sentiment.py | sentiment-based feature generation |
| src/train.py | training and model selection |
| src/evaluate.py | evaluation metrics and comparison |
| src/explain.py | SHAP-based explainability |
| src/gnn_network.py | graph construction, embeddings, and network metrics |
| src/uplift_models.py | uplift modeling and treatment-effect estimation |

---

## 12. Dashboard Features

The Streamlit dashboard provides an interactive experience for users who want to explore predictions and insights without running notebooks.

### Dashboard pages

- Dataset Overview: summary statistics, column information, class balance, sample data
- Exploratory Analysis: churn distribution, tenure/charge visualizations, service patterns
- Churn Prediction: interactive customer profile scoring
- SHAP Explanations: feature importance and interpretability insights
- GNN Insights: graph-based network analysis outputs
- Uplift: treatment recommendations and responder prioritization
- About: project summary and documentation

---

## 13. Outputs Produced by the Project

The project generates several reusable outputs:

- trained models in outputs/models
- evaluation reports in outputs/reports
- figures and charts in outputs/figures
- uplift treatment recommendations in outputs/reports/treatment_recommendations.csv
- GNN analysis outputs and visualizations

Examples of report files include:

- outputs/reports/model_comparison.csv
- outputs/reports/executive_summary.txt
- outputs/reports/findings.md

---

## 14. Explainability Deep Dive

SHAP is used to explain both global and local predictions. This is particularly important in churn modeling because stakeholders often want to understand the reasons behind a risk score rather than just receive a label.

### Why SHAP matters

- highlights the most influential variables
- turns a black-box model into a business-friendly explanation
- supports trust, adoption, and actionability

### What the project explains

- why a customer is predicted as high-risk
- which features contribute most strongly to the outcome
- which retention interventions are most aligned with the customer profile

---

## 15. GNN and Uplift Extensions

### GNN network analysis

The GNN module builds a similarity graph over customers and derives embeddings and network metrics. This allows the project to model structural patterns that simpler tabular approaches may overlook.

### Uplift modeling

The uplift module estimates heterogeneous treatment effects so the project can recommend which customers are likely to respond positively to a retention offer. This is more actionable than simply predicting churn because it supports prioritization of interventions.

---

## 16. Practical Usage Examples

### Example 1: Training and evaluating models

```python
from src.train import train_models
from src.evaluate import evaluate_models

# Example workflow
# train_models(...)
# evaluate_models(...)
```

### Example 2: Loading a saved model

```python
import pickle

with open("outputs/models/best_model.pkl", "rb") as f:
    model = pickle.load(f)
```

### Example 3: Launching the dashboard

```bash
streamlit run dashboard/app.py
```

---

## 17. Future Enhancements

Potential next steps for this project include:

- real-time scoring and deployment
- production APIs for prediction and explanation
- A/B testing of intervention strategies
- more advanced causal inference methods
- automated retraining pipelines
- richer social and network modeling

---

## 18. Troubleshooting

| Issue | Suggested solution |
|-------|--------------------|
| Dataset not found | place the CSV file in data/raw/ |
| NLTK error | run the nltk download command shown above |
| Dashboard fails to start | run the command from the project root or dashboard folder |
| Missing packages | install dependencies from requirements.txt |
| Model files missing | rerun the training notebook or generate outputs again |

---

## 19. References and Libraries

### Libraries used

- pandas
- numpy
- scikit-learn
- xgboost
- shap
- matplotlib
- seaborn
- streamlit
- networkx
- node2vec
- torch / torch-geometric
- econml
- DoWhy

### Useful references

- SHAP documentation
- scikit-learn documentation
- XGBoost documentation
- Streamlit documentation
- NetworkX documentation

---

