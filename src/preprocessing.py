"""
Data Preprocessing Module
Handles data cleaning, missing value treatment, and type conversion.
"""

import logging
from typing import Tuple, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses raw telco churn data for modeling.
    
    Handles:
    - Removal of unnecessary columns
    - Missing value imputation
    - Data type conversion
    - Outlier detection
    """
    
    def __init__(self) -> None:
        """Initialize the preprocessor."""
        self.encoders: dict = {}
        self.column_dtypes: dict = {}
        
    def remove_customer_id(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove customerID column as it's not a predictive feature.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame without customerID
        """
        if 'customerID' in data.columns:
            data = data.drop('customerID', axis=1)
            logger.info("Removed customerID column")
        return data
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        TotalCharges has non-numeric values that appear as spaces.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        # Convert TotalCharges to numeric (handles errors by coercing to NaN)
        if 'TotalCharges' in data.columns:
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
            
            # Fill missing values with median
            if data['TotalCharges'].isnull().sum() > 0:
                median_charges = data['TotalCharges'].median()
                data['TotalCharges'].fillna(median_charges, inplace=True)
                logger.info(f"Imputed {data['TotalCharges'].isnull().sum()} missing TotalCharges with median: {median_charges}")
        
        # Check for remaining missing values
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Remaining missing values:\n{missing_counts[missing_counts > 0]}")
            # Drop rows with missing target variable
            if 'Churn' in data.columns:
                data = data.dropna(subset=['Churn'])
        
        return data
    
    def convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert columns to appropriate data types.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with corrected data types
        """
        # Convert numeric columns
        numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Convert binary columns to category
        binary_cols = [col for col in data.columns 
                      if col not in ['Churn'] and 
                      data[col].nunique() == 2 and 
                      data[col].dtype == 'object']
        for col in binary_cols:
            data[col] = data[col].astype('category')
        
        # Convert other object columns to category
        for col in data.select_dtypes(include=['object']).columns:
            if col != 'Churn':  # Keep Churn as string temporarily
                data[col] = data[col].astype('category')
        
        logger.info("Data types converted")
        self.column_dtypes = data.dtypes.to_dict()
        return data
    
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and flag potential anomalies in numerical features.
        
        Uses IQR method for anomaly detection.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with anomaly information logged
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomalies = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            if len(anomalies) > 0:
                logger.info(f"Found {len(anomalies)} anomalies in {col}")
        
        return data
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps.
        
        Args:
            data: Raw input DataFrame
            
        Returns:
            Cleaned and processed DataFrame
        """
        logger.info("Starting preprocessing pipeline")
        
        data = data.copy()
        data = self.remove_customer_id(data)
        data = self.handle_missing_values(data)
        data = self.convert_data_types(data)
        data = self.detect_anomalies(data)
        
        logger.info(f"Preprocessing complete. Final shape: {data.shape}")
        return data


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to preprocess data.
    
    Args:
        data: Raw DataFrame
        
    Returns:
        Processed DataFrame
    """
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess(data)
