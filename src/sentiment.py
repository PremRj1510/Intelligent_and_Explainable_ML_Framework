"""
Sentiment Analysis Module
Generates synthetic customer feedback and extracts sentiment features using VADER.
"""

import logging
import random
from typing import List, Tuple

import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Generates synthetic customer feedback and analyzes sentiment.
    
    Since the Telco dataset doesn't include text feedback, we create
    realistic synthetic feedback based on customer characteristics
    to demonstrate sentiment as a feature.
    """
    
    # Positive feedback samples
    POSITIVE_FEEDBACK = [
        "Great service quality and fast internet",
        "Excellent customer support team",
        "Very happy with my service",
        "Great value for money",
        "Reliable and fast connection",
        "Best telecom provider in the area",
        "Wonderful experience with billing support",
        "Amazing network coverage",
        "Satisfied with all services offered",
        "Highly recommend this provider"
    ]
    
    # Negative feedback samples
    NEGATIVE_FEEDBACK = [
        "Poor network coverage in my area",
        "Frequent call drops and disconnections",
        "Billing issues and overcharges",
        "Terrible customer service experience",
        "Slow internet speeds despite high charges",
        "Unfair contract terms",
        "Always getting disconnected",
        "Overpriced service",
        "Very disappointed with service",
        "Never resolves my complaints"
    ]
    
    # Neutral feedback samples
    NEUTRAL_FEEDBACK = [
        "Service is okay for the price",
        "Average experience so far",
        "Nothing special but adequate",
        "Similar to other providers",
        "Decent service overall",
        "Service meets expectations",
        "Just average telecom experience",
        "Fair service quality"
    ]
    
    def __init__(self) -> None:
        """Initialize VADER sentiment analyzer."""
        self.sia = SentimentIntensityAnalyzer()
        logger.info("SentimentAnalyzer initialized with VADER")
    
    def generate_synthetic_feedback(self, 
                                   data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic customer feedback based on customer characteristics.
        
        Logic:
        - High churn risk (low tenure, high charges) -> Negative/Neutral
        - Low churn risk (high tenure) -> Positive/Neutral
        - Neutral cases -> Mixed feedback
        
        Args:
            data: Customer data DataFrame
            
        Returns:
            DataFrame with synthetic_feedback column added
        """
        feedback_list = []
        
        for idx, row in data.iterrows():
            # Normalize tenure and charges for probability
            tenure_normalized = row['tenure'] / data['tenure'].max()
            charges_normalized = row['MonthlyCharges'] / data['MonthlyCharges'].max()
            
            # Calculate propensity for positive feedback
            # High tenure → more positive
            # High charges with low tenure → more negative
            positive_propensity = tenure_normalized * 0.7 - charges_normalized * 0.3
            
            rand = random.random()
            
            if positive_propensity > 0.5:
                feedback = random.choice(self.POSITIVE_FEEDBACK)
            elif positive_propensity < -0.5:
                feedback = random.choice(self.NEGATIVE_FEEDBACK)
            else:
                feedback = random.choice(self.NEUTRAL_FEEDBACK)
            
            feedback_list.append(feedback)
        
        data['synthetic_feedback'] = feedback_list
        logger.info(f"Generated synthetic feedback for {len(data)} customers")
        return data
    
    def analyze_sentiment(self, feedback_text: str) -> dict:
        """
        Analyze sentiment of feedback using VADER.
        
        Args:
            feedback_text: Customer feedback text
            
        Returns:
            Dictionary with sentiment scores (compound, pos, neu, neg)
        """
        scores = self.sia.polarity_scores(feedback_text)
        return scores
    
    def extract_sentiment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract sentiment scores for all customer feedback.
        
        Args:
            data: DataFrame with synthetic_feedback column
            
        Returns:
            DataFrame with sentiment_score column added
        """
        if 'synthetic_feedback' not in data.columns:
            logger.warning("No synthetic_feedback column found")
            data = self.generate_synthetic_feedback(data)
        
        sentiment_scores = []
        
        for feedback in data['synthetic_feedback']:
            scores = self.analyze_sentiment(feedback)
            # Use compound score: ranges from -1 (most negative) to 1 (most positive)
            # Normalize to 0-1 range for features
            normalized_score = (scores['compound'] + 1) / 2
            sentiment_scores.append(normalized_score)
        
        data['sentiment_score'] = sentiment_scores
        
        logger.info(f"Extracted sentiment scores")
        logger.info(f"Sentiment score range: [{data['sentiment_score'].min():.3f}, {data['sentiment_score'].max():.3f}]")
        
        return data
    
    def get_sentiment_statistics(self, data: pd.DataFrame) -> dict:
        """
        Get statistics about sentiment scores.
        
        Args:
            data: DataFrame with sentiment_score column
            
        Returns:
            Dictionary with sentiment statistics
        """
        if 'sentiment_score' not in data.columns:
            return {}
        
        stats = {
            'mean': data['sentiment_score'].mean(),
            'std': data['sentiment_score'].std(),
            'min': data['sentiment_score'].min(),
            'max': data['sentiment_score'].max(),
            'median': data['sentiment_score'].median(),
        }
        
        return stats


def generate_sentiment_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to generate sentiment features.
    
    Args:
        data: Customer DataFrame
        
    Returns:
        DataFrame with sentiment_score feature added
    """
    analyzer = SentimentAnalyzer()
    data = analyzer.generate_synthetic_feedback(data)
    data = analyzer.extract_sentiment_features(data)
    
    logger.info("Sentiment Analysis Summary:")
    logger.info(analyzer.get_sentiment_statistics(data))
    
    return data
