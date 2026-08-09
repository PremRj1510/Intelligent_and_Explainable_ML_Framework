import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import preprocess_data
from src.sentiment import generate_sentiment_features
from src.feature_engineering import FeatureEngineer

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "best_model.pkl"
ENGINEERED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "engineered_data.csv"
GNN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "engineered_data_with_gnn.csv"
GNN_IMAGE_PATH = PROJECT_ROOT / "outputs" / "figures" / "18_gnn_network_analysis.png"
UPLIFT_IMAGE_1_PATH = PROJECT_ROOT / "outputs" / "figures" / "19_uplift_cate_distributions.png"
UPLIFT_IMAGE_2_PATH = PROJECT_ROOT / "outputs" / "figures" / "20_uplift_roi_analysis.png"
UPLIFT_RECOMMENDATIONS_PATH = PROJECT_ROOT / "outputs" / "reports" / "treatment_recommendations.csv"



# Configure page
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

pages = {
    "📈 Dataset Overview": "page_overview",
    "🔍 Exploratory Analysis": "page_eda",
    "🎯 Churn Prediction": "page_prediction",
    "💡 SHAP Explanations": "page_shap",
    "🕸️ GNN Insights": "page_gnn",
    "🎁 Uplift": "page_uplift",
    "📚 About": "page_about"
}


def render_page_title(title: str):
    """Render a centered page title at the top of the dashboard."""
    st.markdown(
        f"<h1 style='text-align: center; margin-bottom: 0.4rem;'>{title}</h1>",
        unsafe_allow_html=True,
    )


def resolve_project_path(*parts):
    """Resolve a path relative to the project root, with a cwd fallback."""
    candidates = [PROJECT_ROOT.joinpath(*parts), Path.cwd().joinpath(*parts)]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

@st.cache_resource
def load_data():
    """Load engineered dataset."""
    path = resolve_project_path("data", "processed", "engineered_data.csv")
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_model(model_name="best_model"):
    """Load trained model."""
    path = resolve_project_path("outputs", "models", f"{model_name}.pkl")
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_all_models():
    """Load all trained models."""
    models = {}
    model_dir = resolve_project_path("outputs", "models")
    
    if model_dir.exists():
        for model_file in model_dir.glob("*.pkl"):
            if model_file.name != "best_model.pkl":
                with open(model_file, 'rb') as f:
                    model_name = model_file.stem.replace('_', ' ').title()
                    models[model_name] = pickle.load(f)
    
    return models

@st.cache_resource
def load_training_assets():
    raw_df = pd.read_csv(RAW_DATA_PATH)
    cleaned = preprocess_data(raw_df.copy())
    with_sentiment = generate_sentiment_features(cleaned.copy())

    feature_engineer = FeatureEngineer()
    engineered = feature_engineer.engineer_features(with_sentiment.copy(), fit=True)

    model = pickle.load(open(MODEL_PATH, "rb"))

    training_stats = {
        "tenure_max": float(engineered["tenure"].max()),
        "monthly_charge_max": float(engineered["MonthlyCharges"].max()),
        "service_count_max": float(engineered["service_count"].max()) if "service_count" in engineered.columns else 1.0,
        "monthly_charge_q25": float(engineered["MonthlyCharges"].quantile(0.25)),
        "monthly_charge_q50": float(engineered["MonthlyCharges"].quantile(0.50)),
        "monthly_charge_q75": float(engineered["MonthlyCharges"].quantile(0.75)),
    }

    expected_columns = [col for col in engineered.drop(columns=["Churn"]).columns]

    return feature_engineer, model, engineered, expected_columns, training_stats


@st.cache_data
def load_processed_data():
    return pd.read_csv(ENGINEERED_DATA_PATH)


@st.cache_data
def load_gnn_data():
    return pd.read_csv(GNN_DATA_PATH)


@st.cache_data
def load_treatment_recommendations():
    return pd.read_csv(UPLIFT_RECOMMENDATIONS_PATH)


def transform_customer_to_features(raw_row: dict, feature_engineer: FeatureEngineer, expected_columns, training_stats):
    data = pd.DataFrame([raw_row])

    # Preprocess
    data = preprocess_data(data.copy())

    # Add sentiment features
    data = generate_sentiment_features(data.copy())

    # Tenure features
    data["tenure_phase"] = pd.cut(
        data["tenure"],
        bins=[0, 6, 12, 24, 36, 72],
        labels=["very_new", "new", "established", "loyal", "very_loyal"],
        include_lowest=True,
    )
    data["is_early_phase"] = (data["tenure"] <= 6).astype(int)
    data["tenure_normalized"] = data["tenure"] / training_stats["tenure_max"]

    # Charge features
    data["tenure"] = data["tenure"].replace(0, 1)
    data["charge_per_tenure"] = data["MonthlyCharges"] / (data["tenure"] + 1)
    data["avg_monthly_value"] = data["TotalCharges"] / (data["tenure"] + 1)

    # Quartile-based charge bucket, aligned to training distribution
    bins = [-np.inf, training_stats["monthly_charge_q25"], training_stats["monthly_charge_q50"], training_stats["monthly_charge_q75"], np.inf]
    labels = ["q1_low", "q2_med_low", "q3_med_high", "q4_high"]
    data["monthly_charge_quartile"] = pd.cut(
        data["MonthlyCharges"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    data["is_high_charge"] = (data["MonthlyCharges"] > training_stats["monthly_charge_q75"]).astype(int)

    # Service features
    service_cols = [
        "PhoneService", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in service_cols:
        if col in data.columns:
            data[f"{col}_adopted"] = (data[col].isin(["Yes", 1])).astype(int)

    adopted_cols = [f"{c}_adopted" for c in service_cols if f"{c}_adopted" in data.columns]
    if adopted_cols:
        data["service_count"] = data[adopted_cols].sum(axis=1)

    # Contract features
    if "Contract" in data.columns:
        contract_risk = {"Month-to-month": 1.0, "One year": 0.5, "Two year": 0.2}
        data["contract_risk_score"] = data["Contract"].map(contract_risk)
        data["is_month_to_month"] = (data["Contract"] == "Month-to-month").astype(int)

    # Customer value score
    tenure_score = data["tenure"] / training_stats["tenure_max"]
    charge_score = data["MonthlyCharges"] / training_stats["monthly_charge_max"]
    service_score = data["service_count"] / training_stats["service_count_max"] if "service_count" in data.columns else 0
    data["customer_value_score"] = tenure_score * 0.5 + charge_score * 0.3 + service_score * 0.2

    # Encode categorical features using encoders fit on training data
    data = feature_engineer.encode_categorical_features(data, fit=False)

    # Align columns to the training feature order
    data = data.reindex(columns=expected_columns, fill_value=0)

    return data

# ============================================================================
# PAGE 1: DATASET OVERVIEW
# ============================================================================

def page_overview():
    render_page_title("📈 Dataset Overview")
    
    data = load_data()
    if data is None:
        st.error("Dataset not found. Please run feature engineering notebook first.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(data):,}")
    with col2:
        st.metric("Features", f"{data.shape[1]}")
    with col3:
        churn_rate = (data['Churn'] == 1).mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    with col4:
        st.metric("Churn Count", f"{(data['Churn'] == 1).sum():,}")
    
    st.divider()
    
    # Data Summary Statistics
    st.subheader("Data Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape**")
        st.write(f"Rows: {data.shape[0]:,}")
        st.write(f"Columns: {data.shape[1]}")
        
        st.write("**Feature Types**")
        st.write(f"Numeric: {data.select_dtypes(include=[np.number]).shape[1]}")
        st.write(f"Categorical: {data.select_dtypes(include=['object', 'category']).shape[1]}")
    
    with col2:
        st.write("**Missing Values**")
        missing = data.isnull().sum().sum()
        st.write(f"Total missing: {missing}")
        
        st.write("**Churn Distribution**")
        churn_dist = data['Churn'].value_counts()
        st.write(f"No Churn: {churn_dist[0]:,} ({churn_dist[0]/len(data)*100:.1f}%)")
        st.write(f"Churn: {churn_dist[1]:,} ({churn_dist[1]/len(data)*100:.1f}%)")
    
    st.divider()
    
    # Display sample data
    st.subheader("Sample Data")
    st.dataframe(data.head(10), use_container_width=True)


# ============================================================================
# PAGE 2: EXPLORATORY ANALYSIS
# ============================================================================

def page_eda():
    render_page_title("🔍 Exploratory Data Analysis")
    
    data = load_data()
    if data is None:
        st.error("Dataset not found.")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["Churn Distribution", "Tenure Analysis", "Charges Analysis", "Service Analysis"])
    
    with tab1:
        st.subheader("Churn Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots()
            churn_counts = data['Churn'].value_counts()
            ax.pie(churn_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%',
                   colors=['#2ecc71', '#e74c3c'], startangle=90)
            ax.set_title('Churn Distribution', fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots()
            churn_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
            ax.set_title('Churn Count', fontweight='bold')
            ax.set_xlabel('Churn Status')
            ax.set_ylabel('Number of Customers')
            ax.set_xticklabels(['No Churn', 'Churn'], rotation=0)
            st.pyplot(fig)
    
    with tab2:
        st.subheader("Tenure Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots()
            data.boxplot(column='tenure', by='Churn', ax=ax)
            ax.set_title('Tenure by Churn Status', fontweight='bold')
            ax.set_xlabel('Churn')
            ax.set_ylabel('Tenure (months)')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots()
            for churn in [0, 1]:
                label = 'Churn' if churn == 1 else 'No Churn'
                color = '#e74c3c' if churn == 1 else '#2ecc71'
                ax.hist(data[data['Churn'] == churn]['tenure'], alpha=0.6, 
                       label=label, bins=30, color=color)
            ax.set_title('Tenure Distribution', fontweight='bold')
            ax.set_xlabel('Tenure (months)')
            ax.set_ylabel('Frequency')
            ax.legend()
            st.pyplot(fig)
    
    with tab3:
        st.subheader("Monthly Charges Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots()
            data.boxplot(column='MonthlyCharges', by='Churn', ax=ax)
            ax.set_title('Monthly Charges by Churn', fontweight='bold')
            ax.set_xlabel('Churn')
            ax.set_ylabel('Monthly Charges ($)')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots()
            for churn in [0, 1]:
                label = 'Churn' if churn == 1 else 'No Churn'
                color = '#e74c3c' if churn == 1 else '#2ecc71'
                charges = data[data['Churn'] == churn]['MonthlyCharges']
                ax.hist(charges, bins=30, alpha=0.6, label=label, color=color)
            ax.set_title('Monthly Charges Distribution', fontweight='bold')
            ax.set_xlabel('Monthly Charges ($)')
            ax.set_ylabel('Frequency')
            ax.legend()
            st.pyplot(fig)
    
    with tab4:
        st.subheader("Service Adoption Analysis")
        
        if 'service_count' in data.columns:
            fig, ax = plt.subplots()
            service_churn = pd.crosstab(data['service_count'], data['Churn'], normalize='index') * 100
            service_churn[1].plot(kind='bar', ax=ax, color='#3498db')
            ax.set_title('Churn Rate by Service Count', fontweight='bold')
            ax.set_xlabel('Number of Services')
            ax.set_ylabel('Churn Rate (%)')
            st.pyplot(fig)

def page_prediction():
    render_page_title("🎯 Real churn inference")
    st.write("This view uses the same preprocessing and feature-engineering path as the training notebooks, then scores the result with the saved best model.")

    feature_engineer, model, engineered_df, expected_columns, training_stats = load_training_assets()

    with st.form("customer_form"):
        st.subheader("Customer profile")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.checkbox("Senior citizen")
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            phone_service = st.selectbox("Phone service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet service", ["No", "DSL", "Fiber optic"])
            online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])

        with col3:
            device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        monthly_charges = st.slider("Monthly charges", 18.0, 120.0, 70.0)
        total_charges = st.slider("Total charges", 20.0, 9000.0, 800.0)
        paperless_billing = st.selectbox("Paperless billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

        submitted = st.form_submit_button("Run prediction")

    if submitted:
        raw_row = {
            "customerID": "demo_customer",
            "gender": gender,
            "SeniorCitizen": int(senior_citizen),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": "No",
        }

        features = transform_customer_to_features(raw_row, feature_engineer, expected_columns, training_stats)
        proba = model.predict_proba(features)[0, 1]
        predicted_label = "Likely churn" if proba >= 0.5 else "Likely stays"

        st.metric("Churn probability", f"{proba * 100:.1f}%")
        st.metric("Prediction", predicted_label)

        st.subheader("Engineered features used by the model")
        st.dataframe(features.head(1), use_container_width=True)

def page_shap():
    render_page_title("💡 Model Explainability (SHAP)")
    
    st.write("""
    SHAP (SHapley Additive exPlanations) values show how much each feature 
    contributes to the model's prediction, enabling interpretable AI.
    """)
    
    st.divider()
    
    st.subheader("🔴 Key Churn Drivers (in order of importance)")
    
    drivers = [
        ("1. Tenure", "Strongest predictor. New customers (0-6 months) have 5x higher churn."),
        ("2. Contract Type", "Month-to-month contracts have 42% churn vs 3% for 2-year."),
        ("3. Service Count", "More services = higher switching costs. Each service reduces churn risk."),
        ("4. Monthly Charges", "Non-linear: very high and very low both increase churn."),
        ("5. Sentiment Score", "Customer satisfaction provides real-time churn signal."),
    ]
    
    for title, description in drivers:
        st.write(f"**{title}**")
        st.write(description)
        st.write("")
    
    st.divider()
    
    st.subheader("💼 Business Insights")
    
    insights = {
        "Early Phase (0-6 months)": "27% churn rate. Intensive onboarding + support crucial.",
        "Service Adoption": "Bundle 3-5 key services to increase switching costs.",
        "Contract Conversion": "Incentivize longer terms (10-20% discount for 1-year, 20-25% for 2-year).",
        "Pricing Strategy": "$30-$70/month optimal. Both higher and lower charges increase churn.",
        "Sentiment Monitoring": "Negative feedback + short tenure = 60% churn probability.",
    }
    
    for insight, detail in insights.items():
        st.write(f"**{insight}**")
        st.write(detail)
        st.write("")
    
    st.divider()
    
    st.subheader("📊 Impact Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Churn Rate", "26%")
    with col2:
        st.metric("Addressable via interventions", "4-6%")
    with col3:
        st.metric("Potential Revenue Impact", "$250K-500K/year")


# ============================================================================
# PAGE 5: ABOUT
# ============================================================================

def page_gnn():
    render_page_title("🕸️ GNN network insights")
    st.write("This section shows the graph-based analysis generated in the GNN notebook.")

    if GNN_IMAGE_PATH.exists():
        st.image(str(GNN_IMAGE_PATH), caption="Customer network graph and centrality analysis")
    else:
        st.warning("GNN image not found.")

    gnn_df = load_gnn_data()
    if gnn_df is not None:
        st.subheader("GNN-enhanced dataset preview")
        st.dataframe(gnn_df.head(10), use_container_width=True)


def page_uplift():
    render_page_title("🎁 Uplift recommendations")
    st.write("These outputs come from the uplift modeling notebook and highlight which customers are most likely to respond to a retention offer.")

    col1, col2 = st.columns(2)
    with col1:
        if UPLIFT_IMAGE_1_PATH.exists():
            st.image(str(UPLIFT_IMAGE_1_PATH), caption="CATE distributions")
        else:
            st.warning("CATE distribution image not found.")
    with col2:
        if UPLIFT_IMAGE_2_PATH.exists():
            st.image(str(UPLIFT_IMAGE_2_PATH), caption="ROI analysis")
        else:
            st.warning("ROI analysis image not found.")

    recommendations = load_treatment_recommendations()
    if recommendations is not None:
        st.subheader("Top treatment recommendations")
        st.dataframe(recommendations.head(15), use_container_width=True)

# ============================================================================
# PAGE 5: ABOUT
# ============================================================================

def page_about():
    render_page_title("📚 About This Project")
    
    st.write("""
    ## Explainable ML Framework for Telecom Churn Prediction
    
    ### Project Overview
    This dashboard presents an end-to-end machine learning framework for predicting 
    and explaining customer churn in the telecom industry. The project combines 
    advanced ML models with explainability techniques (SHAP) to enable data-driven 
    retention strategies.
    
    ### Dataset
    - **Source**: Telco Customer Churn (publicly available)
    - **Records**: ~7,043 customers
    - **Features**: 42 engineered features (from raw 20 features)
    - **Target**: Customer Churn (Yes/No)
    
    ### Models Used
    - Logistic Regression (baseline, interpretable)
    - Random Forest (ensemble, feature importance)
    - XGBoost (gradient boosting, best performance)
    
    ### Performance
    - **ROC-AUC**: 0.82-0.85 (all models)
    - **Accuracy**: 80%+
    - **Precision**: 65%+
    - **Recall**: 55%+
    
    ### Key Technologies
    - Python 3.11+
    - scikit-learn, XGBoost
    - SHAP for explainability
    - graph neural networks (GNN) for network analysis
    - Uplift modeling for treatment effect estimation         
    - Streamlit for dashboard
    - Pandas, NumPy for data processing
    
    ### Project Structure
    ```
    project/
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── notebooks/
    │   ├── 01_eda.ipynb
    │   ├── 02_feature_engineering.ipynb
    │   ├── 03_model_training.ipynb
    │   └── 04_explainability.ipynb
    │   ├── 05_gnn_network_analysis.ipynb
    │   └── 06_uplift_modeling.ipynb    
    ├── src/
    │   ├── data_loader.py
    │   ├── preprocessing.py
    │   ├── feature_engineering.py
    │   ├── sentiment.py
    │   ├── train.py
    │   ├── evaluate.py
    │   └── explain.py
    │   ├── gnn_network.py
    │   └── uplift_models.py         
    ├── outputs/
    │   ├── figures/
    │   ├── models/
    │   └── reports/
    └── dashboard/
        └── app.py
    
    ### Contact & Support
    For questions or issues, please refer to the project documentation.
    
    ### License
    This project is for educational and research purposes.
    """)


def main():
     # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", list(pages.keys()))
    
    # Render selected page
    if page == "📈 Dataset Overview":
        page_overview()
    elif page == "🔍 Exploratory Analysis":
        page_eda()
    elif page == "🎯 Churn Prediction":
        page_prediction()
    elif page == "💡 SHAP Explanations":
        page_shap()
    elif page == "🕸️ GNN Insights":
        page_gnn()
    elif page == "🎁 Uplift":
        page_uplift()
    elif page == "📚 About":
        page_about()
    
    # Footer
    st.sidebar.divider()
    st.sidebar.write("---")
    st.sidebar.write("**Explainable ML for Churn Prediction**")
    st.sidebar.write("Built with Python, scikit-learn, and Streamlit")


if __name__ == "__main__":
    main()
