"""
Model Training Module
Trains and evaluates multiple ML models for churn prediction.
"""

import logging
from typing import Dict, Tuple, Any
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class ChurnModelTrainer:
    """
    Trains multiple machine learning models for churn prediction.
    
    Supports:
    - Logistic Regression
    - Random Forest
    - XGBoost
    
    Implements stratified train/test split and cross-validation.
    """
    
    def __init__(self, X_train: pd.DataFrame = None, y_train: pd.Series = None, 
                 test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Initialize the trainer.
        
        Args:
            X_train: Training features (optional)
            y_train: Training target (optional)
            test_size: Proportion for test set
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.X_train: pd.DataFrame = X_train
        self.X_test: pd.DataFrame = None
        self.y_train: pd.Series = y_train
        self.y_test: pd.Series = None
        self.feature_names: list = []
        
        if X_train is not None:
            self.feature_names = X_train.columns.tolist()
    
    def prepare_data(self, 
                    X: pd.DataFrame, 
                    y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare and split data into train/test sets.
        
        Uses stratified split to maintain churn distribution.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.feature_names = X.columns.tolist()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Data split - Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        logger.info(f"Train churn rate: {self.y_train.mean():.3f}")
        logger.info(f"Test churn rate: {self.y_test.mean():.3f}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_logistic_regression(self) -> LogisticRegression:
        """
        Train Logistic Regression model.
        
        Returns:
            Trained LogisticRegression model
        """
        logger.info("Training Logistic Regression...")
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            n_jobs=-1,
            solver='lbfgs'
        )
        
        # Cross-validation
        cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                    cv=5, scoring='roc_auc')
        logger.info(f"Logistic Regression CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train on full training set
        model.fit(self.X_train, self.y_train)
        
        self.models['Logistic Regression'] = model
        logger.info("Logistic Regression training complete")
        
        return model
    
    def train_random_forest(self) -> RandomForestClassifier:
        """
        Train Random Forest model with hyperparameter tuning.
        
        Returns:
            Trained RandomForestClassifier model
        """
        logger.info("Training Random Forest...")
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
        
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        logger.info(f"Best Random Forest params: {grid_search.best_params_}")
        logger.info(f"Best CV AUC: {grid_search.best_score_:.4f}")
        
        best_model = grid_search.best_estimator_
        self.models['Random Forest'] = best_model
        logger.info("Random Forest training complete")
        
        return best_model
    
    def train_xgboost(self) -> XGBClassifier:
        """
        Train XGBoost model with hyperparameter tuning.
        
        Returns:
            Trained XGBClassifier model
        """
        logger.info("Training XGBoost...")
        
        # Hyperparameter grid
        param_grid = {
            'max_depth': [5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200],
            'subsample': [0.8, 0.9]
        }
        
        xgb = XGBClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
            eval_metric='logloss'
        )
        
        grid_search = GridSearchCV(
            xgb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        logger.info(f"Best XGBoost params: {grid_search.best_params_}")
        logger.info(f"Best CV AUC: {grid_search.best_score_:.4f}")
        
        best_model = grid_search.best_estimator_
        self.models['XGBoost'] = best_model
        logger.info("XGBoost training complete")
        
        return best_model
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        Train all models.
        
        Returns:
            Dictionary of trained models
        """
        logger.info("=" * 60)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 60)
        
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        
        logger.info(f"Trained {len(self.models)} models")
        return self.models
    
    def get_model(self, name: str) -> Any:
        """Get a trained model by name."""
        return self.models.get(name)
    
    def save_models(self, output_dir: str) -> None:
        """
        Save trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = output_path / f"{name.lower().replace(' ', '_')}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {name} to {filepath}")
    
    def load_models(self, model_dir: str) -> None:
        """
        Load trained models from disk.
        
        Args:
            model_dir: Directory containing model files
        """
        model_path = Path(model_dir)
        
        for filepath in model_path.glob("*.pkl"):
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
                model_name = filepath.stem.replace('_', ' ').title()
                self.models[model_name] = model
            logger.info(f"Loaded {model_name} from {filepath}")


def train_models(X: pd.DataFrame, 
                y: pd.Series,
                output_dir: str = None) -> ChurnModelTrainer:
    """
    Convenience function to train all models.
    
    Args:
        X: Features DataFrame
        y: Target Series
        output_dir: Optional directory to save models
        
    Returns:
        ChurnModelTrainer instance with trained models
    """
    trainer = ChurnModelTrainer()
    trainer.prepare_data(X, y)
    trainer.train_all_models()
    
    if output_dir:
        trainer.save_models(output_dir)
    
    return trainer
