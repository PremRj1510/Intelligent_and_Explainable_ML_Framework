"""
Explainability Module
Implements SHAP-based explanations for model predictions.
"""

import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    Provides explainable AI using SHAP (SHapley Additive exPlanations).
    
    Generates:
    - Global explanations (summary plots, feature importance)
    - Local explanations (waterfall plots, force plots)
    - Prediction interpretation
    """
    
    def __init__(self, model: Any, X_train: pd.DataFrame, feature_names: list = None) -> None:
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model to explain
            X_train: Training data for SHAP background
            feature_names: Optional list of feature names
        """
        self.model = model
        self.X_train = X_train
        self.explanation_data = None
        self.feature_names = feature_names or X_train.columns.tolist()
        
        # Create SHAP explainer (TreeExplainer for tree models, KernelExplainer for others)
        model_type = type(model).__name__
        
        if model_type in ['RandomForestClassifier', 'XGBClassifier']:
            self.explainer = shap.TreeExplainer(model)
            logger.info(f"Created TreeExplainer for {model_type}")
        else:
            # Use a sample of training data as background for efficiency
            background = shap.sample(X_train, min(100, len(X_train)))
            self.explainer = shap.KernelExplainer(model.predict_proba, background)
            logger.info(f"Created KernelExplainer for {model_type}")
        
        self.shap_values = None
        self.base_value = None
    
    def calculate_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for a dataset.
        
        Args:
            X: Features to explain
            
        Returns:
            SHAP values array
        """
        logger.info("Calculating SHAP values...")
        self.explanation_data = X.copy()
        self.shap_values = self.explainer.shap_values(X)
        
        # For binary classification, get positive class SHAP values
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        self.base_value = self.explainer.expected_value
        if isinstance(self.base_value, list):
            self.base_value = self.base_value[1]
        
        logger.info(f"Calculated SHAP values. Shape: {self.shap_values.shape}")
        return self.shap_values
    
    def plot_summary(self, output_path: str = None) -> None:
        """
        Plot SHAP summary plot (global feature importance).
        
        Args:
            output_path: Optional path to save figure
        """
        if self.shap_values is None:
            logger.warning("SHAP values not calculated")
            return
        
        explanation_data = self.explanation_data if self.explanation_data is not None else self.X_train
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, explanation_data,
                         feature_names=self.feature_names,
                         show=False)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP summary plot to {output_path}")
        
        plt.close()
    
    def plot_bar(self, output_path: str = None) -> None:
        """
        Plot SHAP bar plot of mean absolute SHAP values.
        
        Args:
            output_path: Optional path to save figure
        """
        if self.shap_values is None:
            logger.warning("SHAP values not calculated")
            return
        
        explanation_data = self.explanation_data if self.explanation_data is not None else self.X_train
        plt.figure(figsize=(10, 6))
        shap.bar_plot(self.shap_values, explanation_data,
                     feature_names=self.feature_names,
                     show=False)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP bar plot to {output_path}")
        
        plt.close()
    
    def plot_waterfall(self, idx: int, output_path: str = None) -> None:
        """
        Plot SHAP waterfall plot for a single prediction.
        
        Shows how features contribute to pushing prediction from base value.
        
        Args:
            idx: Index of instance to explain
            output_path: Optional path to save figure
        """
        if self.shap_values is None:
            logger.warning("SHAP values not calculated")
            return
        
        explanation_data = self.explanation_data if self.explanation_data is not None else self.X_train
        explanation = shap.Explanation(
            values=self.shap_values[idx],
            base_values=self.base_value,
            data=explanation_data.iloc[idx],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP waterfall plot to {output_path}")
        
        plt.close()
    
    def explain_prediction(self, instance: pd.DataFrame) -> Dict[str, Any]:
        """
        Provide detailed explanation for a single prediction.
        
        Args:
            instance: A single row from the dataset, as a DataFrame.
            
        Returns:
            Dictionary with explanation details
        """
        # Get SHAP values for this instance
        shap_vals_raw = self.explainer.shap_values(instance)
        
        # Handle different SHAP value formats from different explainers
        if isinstance(shap_vals_raw, list):
            # This is the case for multi-class classification, shap_vals is a list of arrays.
            # We take the values for the positive class (class 1).
            shap_values_for_prediction = shap_vals_raw[1]
        elif len(shap_vals_raw.shape) == 3:
            # This is for binary classifiers that return a 3D array (n_samples, n_features, n_classes)
            # We take the values for the positive class (class 1).
            shap_values_for_prediction = shap_vals_raw[:, :, 1]
        else:
            # This is for regressors or binary classifiers that return a 2D array (n_samples, n_features)
            shap_values_for_prediction = shap_vals_raw
        
        # We passed a single instance, so we take the first row of shap values.
        shap_vals = shap_values_for_prediction[0]
        
        # Get top contributing features
        feature_importance = list(zip(self.feature_names, shap_vals))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Get prediction probability
        pred_proba = self.model.predict_proba(instance)[0]
        
        explanation = {
            'churn_probability': pred_proba[1],
            'no_churn_probability': pred_proba[0],
            'base_value': self.base_value,
            'top_features': feature_importance[:5],
            'positive_contributors': [(f, v) for f, v in feature_importance if v > 0][:3],
            'negative_contributors': [(f, v) for f, v in feature_importance if v < 0][:3],
        }
        
        return explanation
    
    def generate_report(self, X_explain: pd.DataFrame, output_dir: str) -> str:
        """
        Generate comprehensive SHAP explanation report.
        
        Args:
            X_explain: Dataset to explain
            output_dir: Directory to save report and figures
            
        Returns:
            Path to generated report
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate SHAP values
        self.calculate_shap_values(X_explain)
        
        # Generate plots
        self.plot_summary(f"{output_dir}/shap_summary.png")
        self.plot_bar(f"{output_dir}/shap_feature_importance.png")
        
        # Sample explanations
        for i in [0, len(X_explain) // 2, len(X_explain) - 1]:
            self.plot_waterfall(i, f"{output_dir}/shap_waterfall_sample_{i}.png")
        
        # Generate text report
        report_path = f"{output_dir}/shap_report.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SHAP EXPLAINABILITY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("GLOBAL EXPLANATIONS\n")
            f.write("-" * 80 + "\n")
            f.write("The SHAP summary plot shows the impact of each feature on the model output.\n")
            f.write("Red points indicate high feature values contributing to churn prediction.\n")
            f.write("Blue points indicate low feature values.\n\n")
            
            f.write("TOP 10 MOST IMPORTANT FEATURES\n")
            f.write("-" * 80 + "\n")
            mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
            top_features = sorted(zip(self.feature_names, mean_abs_shap), 
                                key=lambda x: x[1], reverse=True)[:10]
            
            for rank, (feature, importance) in enumerate(top_features, 1):
                f.write(f"{rank:2d}. {feature:30s} - Impact: {importance:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("INTERPRETATION\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Key Insights:\n")
            f.write("1. Tenure is typically the strongest predictor (negative for churn)\n")
            f.write("2. Contract type significantly influences churn (month-to-month = higher risk)\n")
            f.write("3. Monthly charges show non-linear relationship with churn\n")
            f.write("4. Sentiment score provides additional predictive signal\n")
            f.write("5. Service adoption (multiple services) reduces churn risk\n")
        
        logger.info(f"Generated SHAP report at {report_path}")
        return report_path


def explain_models(models: Dict[str, Any],
                  X_train: pd.DataFrame,
                  X_test: pd.DataFrame,
                  output_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Generate SHAP explanations for all models.
    
    Args:
        models: Dictionary of trained models
        X_train: Training data
        X_test: Test data for explanation
        output_dir: Directory to save explanation outputs
        
    Returns:
        Dictionary of explanations per model
    """
    explanations = {}
    
    for model_name, model in models.items():
        logger.info(f"\nGenerating explanations for {model_name}...")
        
        model_output_dir = f"{output_dir}/{model_name.lower().replace(' ', '_')}"
        
        explainer = SHAPExplainer(model, X_train)
        explainer.generate_report(X_test, model_output_dir)
        
        explanations[model_name] = {
            'explainer': explainer,
            'output_dir': model_output_dir
        }
    
    return explanations
