"""
Data Loading Module
Handles data import, validation, and initial exploration for the Telco Churn dataset.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TelcoDataLoader:
    """
    Data loader for Telco customer churn dataset.
    
    Handles loading, validation, and basic exploration of the raw dataset.
    """
    
    def __init__(self, data_path: str) -> None:
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV file containing the dataset
        """
        self.data_path = Path(data_path)
        self.raw_data: Optional[pd.DataFrame] = None
        self.data_info: dict = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from CSV file.
        
        Returns:
            DataFrame with raw data
            
        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        try:
            self.raw_data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.raw_data.shape}")
            return self.raw_data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self) -> bool:
        """
        Validate data integrity and structure.
        
        Returns:
            True if validation passes
        """
        if self.raw_data is None:
            logger.error("No data loaded")
            return False
        
        # Check for required columns
        required_cols = ['customerID', 'Churn']
        if not all(col in self.raw_data.columns for col in required_cols):
            logger.error(f"Missing required columns: {required_cols}")
            return False
        
        logger.info("Data validation passed")
        return True
    
    def get_data_summary(self) -> dict:
        """
        Get summary statistics about the dataset.
        
        Returns:
            Dictionary with data summary information
        """
        if self.raw_data is None:
            return {}
        
        summary = {
            'shape': self.raw_data.shape,
            'columns': list(self.raw_data.columns),
            'dtypes': self.raw_data.dtypes.to_dict(),
            'missing_values': self.raw_data.isnull().sum().to_dict(),
            'numeric_cols': self.raw_data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_cols': self.raw_data.select_dtypes(include=['object']).columns.tolist(),
        }
        
        logger.info(f"Dataset Summary:\n{pd.DataFrame(summary['missing_values'], index=[0])}")
        return summary
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the loaded data.
        
        Returns:
            DataFrame with raw data
        """
        if self.raw_data is None:
            logger.warning("No data loaded. Call load_data() first.")
            return pd.DataFrame()
        return self.raw_data.copy()


def load_telco_data(data_path: str) -> pd.DataFrame:
    """
    Convenience function to load Telco churn data.
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        DataFrame with loaded data
    """
    loader = TelcoDataLoader(data_path)
    data = loader.load_data()
    loader.validate_data()
    logger.info("Data summary:")
    logger.info(loader.get_data_summary())
    return data
