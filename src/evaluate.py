"""
Model Evaluation Module
Evaluates trained models and generates evaluation metrics and visualizations.
"""

import logging
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates machine learning models for churn prediction.
    
    Generates:
    - Performance metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
    - Confusion matrices
    - ROC and Precision-Recall curves
    - Comprehensive model comparison
    """
    
    def __init__(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Initialize evaluator.
        
        Args:
            X_test: Test features
            y_test: Test target values
        """
        self.X_test = X_test
        self.y_test = y_test
        self.results: Dict[str, Dict[str, float]] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.probabilities: Dict[str, np.ndarray] = {}
    
    def evaluate_model(self, model: Any, model_name: str) -> Dict[str, float]:
        """
        Evaluate a single model.
        
        Args:
            model: Trained model
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        self.predictions[model_name] = y_pred
        self.probabilities[model_name] = y_pred_proba
        
        # Metrics
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, zero_division=0),
            'Recall': recall_score(self.y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(self.y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba),
        }
        
        self.results[model_name] = metrics
        
        logger.info(f"\n{model_name} Performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, models: Dict[str, Any]) -> pd.DataFrame:
        """
        Evaluate all models and return comparison DataFrame.
        
        Args:
            models: Dictionary of model_name -> model
            
        Returns:
            DataFrame with comparison of all models
        """
        logger.info("=" * 60)
        logger.info("EVALUATING ALL MODELS")
        logger.info("=" * 60)
        
        for model_name, model in models.items():
            self.evaluate_model(model, model_name)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.results).T
        
        logger.info("\nModel Comparison:")
        logger.info(comparison_df.to_string())
        
        return comparison_df
    
    def plot_confusion_matrix(self, model_name: str, output_path: str = None) -> None:
        """
        Plot confusion matrix for a model.
        
        Args:
            model_name: Name of the model
            output_path: Optional path to save figure
        """
        if model_name not in self.predictions:
            logger.warning(f"No predictions found for {model_name}")
            return
        
        y_pred = self.predictions[model_name]
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {output_path}")
        
        plt.close()
    
    def plot_roc_curve(self, output_path: str = None) -> None:
        """
        Plot ROC curves for all models.
        
        Args:
            output_path: Optional path to save figure
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, y_pred_proba in self.probabilities.items():
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curve to {output_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(self, output_path: str = None) -> None:
        """
        Plot Precision-Recall curves for all models.
        
        Args:
            output_path: Optional path to save figure
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, y_pred_proba in self.probabilities.items():
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            f1 = f1_score(self.y_test, (y_pred_proba > 0.5).astype(int))
            plt.plot(recall, precision, label=f'{model_name} (F1 = {f1:.3f})', linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Model Comparison')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PR curve to {output_path}")
        
        plt.close()
    
    def plot_metrics_comparison(self, output_path: str = None) -> None:
        """
        Plot bar chart comparing all metrics across models.
        
        Args:
            output_path: Optional path to save figure
        """
        comparison_df = pd.DataFrame(self.results).T
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        metrics = comparison_df.columns
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            comparison_df[metric].sort_values().plot(kind='barh', ax=ax)
            ax.set_title(metric)
            ax.set_xlabel('Score')
            ax.set_xlim([0, 1])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved metrics comparison to {output_path}")
        
        plt.close()
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Get model comparison as DataFrame.
        
        Returns:
            DataFrame with all models and their metrics
        """
        return pd.DataFrame(self.results).T


def evaluate_models(models: Dict[str, Any],
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   output_dir: str = None) -> pd.DataFrame:
    """
    Convenience function to evaluate all models.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test target values
        output_dir: Optional directory to save figures
        
    Returns:
        DataFrame with model comparison
    """
    evaluator = ModelEvaluator(X_test, y_test)
    comparison = evaluator.evaluate_all_models(models)
    
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        evaluator.plot_roc_curve(f"{output_dir}/roc_curve.png")
        evaluator.plot_precision_recall_curve(f"{output_dir}/pr_curve.png")
        evaluator.plot_metrics_comparison(f"{output_dir}/metrics_comparison.png")
        
        for model_name in models.keys():
            evaluator.plot_confusion_matrix(
                model_name, 
                f"{output_dir}/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            )
    
    return comparison
