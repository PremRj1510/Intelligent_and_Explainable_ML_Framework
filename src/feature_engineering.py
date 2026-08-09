"""
Feature Engineering Module
Creates advanced features with business justification for churn prediction.
"""

import logging
from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates and transforms features for churn prediction.
    
    Features are created with clear business justification for
    interpretability and explainability.
    """
    
    def __init__(self) -> None:
        """Initialize the feature engineer."""
        self.scalers: dict = {}
        self.encoders: dict = {}
        self.feature_descriptions: dict = {}
    
    def create_tenure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to customer tenure.
        
        Business Justification:
        - Customer lifetime: Long-tenure customers are more stable
        - Early churn risk: New customers have higher churn probability
        
        Args:
            data: Input DataFrame with tenure column
            
        Returns:
            DataFrame with new tenure features
        """
        # Tenure buckets: identify high-risk early phase customers
        data['tenure_phase'] = pd.cut(data['tenure'], 
                                     bins=[0, 6, 12, 24, 36, 72],
                                     labels=['very_new', 'new', 'established', 'loyal', 'very_loyal'],
                                     include_lowest=True)
        
        # Is customer in critical first 6 months?
        data['is_early_phase'] = (data['tenure'] <= 6).astype(int)
        
        # Tenure normalized (0-1 scale)
        data['tenure_normalized'] = data['tenure'] / data['tenure'].max()
        
        self.feature_descriptions['tenure_phase'] = "Customer lifecycle phase"
        self.feature_descriptions['is_early_phase'] = "Binary flag for critical first 6 months"
        self.feature_descriptions['tenure_normalized'] = "Normalized tenure (0-1)"
        
        logger.info("Created tenure features")
        return data
    
    def create_charge_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to charges and costs.
        
        Business Justification:
        - Value perception: Customers paying more may churn if dissatisfied
        - Total investment: Higher total charges indicate loyalty
        - Monthly affordability: Monthly charges relative to tenure
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with new charge features
        """
        # Avoid division by zero
        data['tenure'] = data['tenure'].replace(0, 1)
        
        # Monthly charges relative to tenure (cost per month over customer lifetime)
        data['charge_per_tenure'] = data['MonthlyCharges'] / (data['tenure'] + 1)
        
        # Average value delivered (inverse: high total for long tenure is good)
        data['avg_monthly_value'] = data['TotalCharges'] / (data['tenure'] + 1)
        
        # Monthly charges quartile (relative to others)
        data['monthly_charge_quartile'] = pd.qcut(data['MonthlyCharges'], 
                                                  q=4, 
                                                  labels=['q1_low', 'q2_med_low', 'q3_med_high', 'q4_high'],
                                                  duplicates='drop')
        
        # Is customer in high charge segment?
        q75 = data['MonthlyCharges'].quantile(0.75)
        data['is_high_charge'] = (data['MonthlyCharges'] > q75).astype(int)
        
        self.feature_descriptions['charge_per_tenure'] = "Monthly charges normalized by tenure"
        self.feature_descriptions['avg_monthly_value'] = "Average monthly charge value over tenure"
        self.feature_descriptions['monthly_charge_quartile'] = "Customer's monthly charge quartile"
        self.feature_descriptions['is_high_charge'] = "Flag for high monthly charge customers"
        
        logger.info("Created charge features")
        return data
    
    def create_service_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to services subscribed.
        
        Business Justification:
        - Service stickiness: More services = more switching costs
        - Product engagement: Higher service count indicates active customer
        - Revenue diversity: Multiple services reduce churn risk
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with new service features
        """
        # Count add-on services (Internet, Security, Tech Support, etc.)
        service_cols = [col for col in data.columns 
                       if col in ['PhoneService', 'InternetService', 'OnlineSecurity', 
                                 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                                 'StreamingTV', 'StreamingMovies']]
        
        # Binary count of services (Yes/No conversion)
        service_adoption = 0
        for col in service_cols:
            if col in data.columns:
                data[f'{col}_adopted'] = (data[col].isin(['Yes', 1])).astype(int)
                service_adoption += 1
        
        # Total service count
        if service_adoption > 0:
            adopted_cols = [f'{col}_adopted' for col in service_cols if col in data.columns]
            data['service_count'] = data[adopted_cols].sum(axis=1)
            self.feature_descriptions['service_count'] = "Number of services subscribed"
        
        logger.info("Created service adoption features")
        return data
    
    def create_contract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to contract type and commitment.
        
        Business Justification:
        - Commitment level: Month-to-month has highest churn
        - Lock-in effect: Longer contracts reduce churn
        - Switching cost proxy: Contract length indicates switching cost
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with new contract features
        """
        if 'Contract' in data.columns:
            # Contract risk score (proxy for switching cost)
            contract_risk = {
                'Month-to-month': 1.0,
                'One year': 0.5,
                'Two year': 0.2
            }
            data['contract_risk_score'] = data['Contract'].map(contract_risk)
            
            # Is month-to-month (highest risk)?
            data['is_month_to_month'] = (data['Contract'] == 'Month-to-month').astype(int)
            
            self.feature_descriptions['contract_risk_score'] = "Contract switching cost proxy"
            self.feature_descriptions['is_month_to_month'] = "Flag for month-to-month contracts"
        
        logger.info("Created contract features")
        return data
    
    def create_customer_value_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite customer value score.
        
        Business Justification:
        - Combines tenure, revenue, and engagement
        - High-value customers: stable, revenue-generating, engaged
        - Low-value customers: at-risk, disengaged
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with customer_value_score
        """
        # Components (normalized to 0-1)
        tenure_score = data['tenure'] / data['tenure'].max()
        charge_score = data['MonthlyCharges'] / data['MonthlyCharges'].max()
        
        service_score = 0
        if 'service_count' in data.columns:
            service_score = data['service_count'] / data['service_count'].max()
        
        # Weighted composite (tenure is most important indicator)
        data['customer_value_score'] = (
            tenure_score * 0.5 +  # Stability (50%)
            charge_score * 0.3 +  # Revenue (30%)
            service_score * 0.2   # Engagement (20%)
        )
        
        self.feature_descriptions['customer_value_score'] = "Composite value score (tenure, revenue, engagement)"
        
        logger.info("Created customer value score")
        return data
    
    def encode_categorical_features(self, 
                                   data: pd.DataFrame,
                                   fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            data: Input DataFrame
            fit: If True, fit encoders; if False, transform only
            
        Returns:
            DataFrame with encoded categorical features
        """
        categorical_cols = data.select_dtypes(include=['category', 'object']).columns
        
        for col in categorical_cols:
            if col == 'Churn':
                # Special handling for target
                data[col] = (data[col] == 'Yes').astype(int)
            else:
                if fit:
                    self.encoders[col] = LabelEncoder()
                    data[col] = self.encoders[col].fit_transform(data[col].astype(str))
                else:
                    if col in self.encoders:
                        data[col] = self.encoders[col].transform(data[col].astype(str))
        
        logger.info("Encoded categorical features")
        return data
    
    def engineer_features(self, 
                         data: pd.DataFrame,
                         fit: bool = True) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            data: Input DataFrame
            fit: If True, fit all transformers; if False, transform only
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering pipeline")
        
        data = data.copy()
        
        # Create domain features
        data = self.create_tenure_features(data)
        data = self.create_charge_features(data)
        data = self.create_service_features(data)
        data = self.create_contract_features(data)
        data = self.create_customer_value_score(data)
        
        # Encode categorical features
        data = self.encode_categorical_features(data, fit=fit)
        
        logger.info(f"Feature engineering complete. Shape: {data.shape}")
        logger.info(f"Feature descriptions: {list(self.feature_descriptions.keys())}")
        
        return data
    
    def get_feature_descriptions(self) -> dict:
        """
        Get descriptions of engineered features.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        return self.feature_descriptions


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function for feature engineering.
    
    Args:
        data: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer()
    return engineer.engineer_features(data)
