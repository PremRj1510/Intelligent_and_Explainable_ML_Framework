"""
Uplift Modeling for Heterogeneous Treatment Effect Estimation.

Identifies which customers benefit most from service bundling offers.
Uses multiple meta-learner approaches (T/S/X/R-Learner) for robust CATE estimation.
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UpliftModeler:
    """
    Estimates Conditional Average Treatment Effect (CATE) using multiple meta-learner approaches.
    
    Treatment: Service bundling offer (add complementary services to customer)
    Outcome: Churn probability reduction
    Goal: Identify customers with highest positive treatment effect (best responders)
    
    Methods:
    - T-Learner: Separate models for treatment and control
    - S-Learner: Single model with treatment as feature
    - X-Learner: Cross-fitting method (efficient)
    - R-Learner: Residual-based orthogonal approach
    
    Final: Ensemble CATE from all methods
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize uplift modeler.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.models = {}
        self.random_state = random_state
        self.cate_predictions = {}
        self.logger = logging.getLogger(__name__)
    
    def prepare_treatment_data(
        self,
        df_train: pd.DataFrame,
        treatment_ratio: float = 0.3,
        treatment_effect_magnitude: float = 0.15
    ) -> Dict:
        """
        Prepare treatment/control data for training.
        
        In real-world, this would use historical treatment records.
        Here we simulate based on propensity scoring and treatment effects.
        
        Treatment mechanism:
        - High-propensity = more likely to receive offer
        - Treatment effect heterogeneous: depends on customer profile
        
        Args:
            df_train: Training dataframe
            treatment_ratio: Fraction of customers to assign treatment
            treatment_effect_magnitude: Magnitude of treatment effect
        
        Returns:
            Dictionary with treatment data components
        """
        
        self.logger.info("Preparing treatment/control data...")
        
        # Features (excluding target and ID)
        feature_cols = [col for col in df_train.columns if col not in ['Churn', 'customerID']]
        X = df_train[feature_cols].copy()
        y = df_train['Churn'].values

        # Clean and encode features so downstream models can train without NaN/object issues.
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce')
                if X[col].isna().any():
                    X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna('missing').astype(str)

        X = pd.get_dummies(X, dummy_na=False, dtype=float)
        X = X.fillna(0.0)
        
        # Estimate propensity score (probability of receiving treatment)
        # Based on churn risk: high-risk customers more likely to get offer
        propensity_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        propensity_model.fit(X, y)
        propensity = propensity_model.predict_proba(X)[:, 1]
        
        # Assign treatment with probability proportional to propensity
        treatment = np.random.binomial(1, propensity * treatment_ratio)
        
        self.logger.info(f"Treatment assignment: {treatment.sum()} treated, {(1-treatment).sum()} control")
        
        # Simulate heterogeneous treatment effect
        # Treatment effect depends on customer characteristics
        treatment_effect = np.zeros(len(df_train))
        
        # Effect 1: Early phase customers benefit more from bundling
        if 'tenure_phase' in df_train.columns:
            early_phase_mask = df_train['tenure_phase'] == 'Early Phase'
            treatment_effect[early_phase_mask & (treatment == 1)] = -treatment_effect_magnitude
        
        # Effect 2: Low service count customers benefit from bundling
        if 'service_count' in df_train.columns:
            low_service_mask = df_train['service_count'] <= 2
            treatment_effect[low_service_mask & (treatment == 1)] -= treatment_effect_magnitude * 0.7
        
        # Effect 3: Negative sentiment customers benefit more
        if 'sentiment_score' in df_train.columns:
            negative_sentiment = df_train['sentiment_score'] < 0.3
            treatment_effect[negative_sentiment & (treatment == 1)] -= treatment_effect_magnitude * 0.5
        
        # Create observed outcome: original churn + treatment effect
        outcome = (y + treatment_effect).clip(0, 1)
        
        data_dict = {
            'X': X,
            'y': y,
            'outcome': outcome,
            'treatment': treatment,
            'propensity': propensity,
            'true_effect': treatment_effect
        }
        
        self.logger.info(f"Treatment effect: mean={treatment_effect[treatment==1].mean():.4f}")
        
        return data_dict
    
    def train_tlearner(self, data: Dict) -> np.ndarray:
        """
        T-Learner: Train separate models for treatment and control groups.
        
        CATE = E[Y|X, T=1] - E[Y|X, T=0]
        
        Args:
            data: Dictionary with 'X', 'outcome', 'treatment'
        
        Returns:
            CATE predictions (array)
        """
        
        self.logger.info("Training T-Learner...")
        
        X = data['X']
        outcome = data['outcome']
        treatment = data['treatment']
        
        # Split by treatment group
        X_treat = X[treatment == 1]
        y_treat = outcome[treatment == 1]
        X_control = X[treatment == 0]
        y_control = outcome[treatment == 0]
        
        self.logger.info(f"  Treated group: {len(X_treat)} samples, mean outcome: {y_treat.mean():.4f}")
        self.logger.info(f"  Control group: {len(X_control)} samples, mean outcome: {y_control.mean():.4f}")
        
        # Train separate models
        model_treat = XGBRegressor(
            random_state=self.random_state,
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            verbose=0
        )
        model_control = XGBRegressor(
            random_state=self.random_state,
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            verbose=0
        )
        
        model_treat.fit(X_treat, y_treat)
        model_control.fit(X_control, y_control)
        
        # Predict CATE for all samples
        cate = model_treat.predict(X) - model_control.predict(X)
        
        self.models['TLearner'] = {
            'model_treat': model_treat,
            'model_control': model_control,
            'cate': cate
        }
        
        self.logger.info(f"T-Learner CATE: mean={cate.mean():.4f}, std={cate.std():.4f}")
        return cate
    
    def train_slearner(self, data: Dict) -> np.ndarray:
        """
        S-Learner: Single model with treatment as feature.
        
        CATE = E[Y|X, T=1] - E[Y|X, T=0]
        
        Args:
            data: Dictionary with 'X', 'outcome', 'treatment'
        
        Returns:
            CATE predictions (array)
        """
        
        self.logger.info("Training S-Learner...")
        
        X = data['X']
        outcome = data['outcome']
        treatment = data['treatment']
        
        # Augment features with treatment
        X_augmented = X.copy()
        X_augmented['treatment'] = treatment
        
        # Train single model
        model = XGBRegressor(
            random_state=self.random_state,
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            verbose=0
        )
        model.fit(X_augmented, outcome)
        
        # Predict with T=1 and T=0
        X_treat = X_augmented.copy()
        X_treat['treatment'] = 1
        X_control = X_augmented.copy()
        X_control['treatment'] = 0
        
        cate = model.predict(X_treat) - model.predict(X_control)
        
        self.models['SLearner'] = {
            'model': model,
            'cate': cate
        }
        
        self.logger.info(f"S-Learner CATE: mean={cate.mean():.4f}, std={cate.std():.4f}")
        return cate
    
    def train_xlearner(self, data: Dict) -> np.ndarray:
        """
        X-Learner: Cross-fitting approach for robust CATE.
        
        More statistically efficient than T-Learner.
        
        Args:
            data: Dictionary with 'X', 'outcome', 'treatment', 'propensity'
        
        Returns:
            CATE predictions (array)
        """
        
        self.logger.info("Training X-Learner...")
        
        X = data['X']
        outcome = data['outcome']
        treatment = data['treatment']
        propensity = data['propensity']
        
        # Split data for cross-fitting
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(indices, test_size=0.5, random_state=self.random_state)
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = outcome[train_idx], outcome[test_idx]
        T_train, T_test = treatment[train_idx], treatment[test_idx]
        P_train, P_test = propensity[train_idx], propensity[test_idx]
        
        # Stage 1: Estimate regression models on training fold
        model_1 = XGBRegressor(random_state=self.random_state, max_depth=5, n_estimators=100, verbose=0)
        model_0 = XGBRegressor(random_state=self.random_state, max_depth=5, n_estimators=100, verbose=0)
        
        model_1.fit(X_train[T_train == 1], y_train[T_train == 1])
        model_0.fit(X_train[T_train == 0], y_train[T_train == 0])
        
        # Stage 2: Compute residuals on test fold
        residuals_1 = y_test[T_test == 1] - model_1.predict(X_test[T_test == 1])
        residuals_0 = y_test[T_test == 0] - model_0.predict(X_test[T_test == 0])
        
        # Stage 3: Estimate treatment effect model
        cate_model = XGBRegressor(random_state=self.random_state, max_depth=5, n_estimators=100, verbose=0)
        
        # Combine residuals with treatment indicator
        X_cate = pd.concat([
            X_test[T_test == 1].assign(residual=residuals_1),
            X_test[T_test == 0].assign(residual=residuals_0)
        ])
        y_cate = np.concatenate([
            T_test[T_test == 1].astype(float),
            T_test[T_test == 0].astype(float)
        ])
        
        cate_model.fit(X_cate, y_cate)
        
        # Predict CATE on full data using the same residual feature layout.
        X_full = X.copy()
        X_full['residual'] = np.where(
            treatment == 1,
            outcome - model_1.predict(X),
            outcome - model_0.predict(X)
        )
        cate = cate_model.predict(X_full)
        
        self.models['XLearner'] = {
            'model_1': model_1,
            'model_0': model_0,
            'cate_model': cate_model,
            'cate': cate
        }
        
        self.logger.info(f"X-Learner CATE: mean={cate.mean():.4f}, std={cate.std():.4f}")
        return cate
    
    def train_rlearner(self, data: Dict) -> np.ndarray:
        """
        R-Learner: Residual-based orthogonal approach.
        
        Orthogonalizes treatment and outcome separately.
        
        Args:
            data: Dictionary with 'X', 'outcome', 'treatment', 'propensity'
        
        Returns:
            CATE predictions (array)
        """
        
        self.logger.info("Training R-Learner...")
        
        X = data['X']
        outcome = data['outcome']
        treatment = data['treatment']
        propensity = data['propensity']
        
        # Nuisance: Predict outcome
        m_model = XGBRegressor(random_state=self.random_state, max_depth=5, n_estimators=100, verbose=0)
        m_model.fit(X, outcome)
        m_pred = m_model.predict(X)
        
        # Nuisance: Predict treatment
        e_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        e_model.fit(X, treatment)
        e_pred = e_model.predict_proba(X)[:, 1]
        
        # Orthogonalize: residuals
        residuals_outcome = outcome - m_pred
        residuals_treatment = treatment - e_pred
        
        # Estimate treatment effect on residuals
        cate_model = XGBRegressor(random_state=self.random_state, max_depth=5, n_estimators=100, verbose=0)
        
        # Weighted fit
        weights = residuals_treatment ** 2 + (1 - residuals_treatment) ** 2
        cate_model.fit(X, residuals_outcome, sample_weight=weights)
        
        cate = cate_model.predict(X)
        
        self.models['RLearner'] = {
            'm_model': m_model,
            'e_model': e_model,
            'cate_model': cate_model,
            'cate': cate
        }
        
        self.logger.info(f"R-Learner CATE: mean={cate.mean():.4f}, std={cate.std():.4f}")
        return cate
    
    def ensemble_cate(self) -> np.ndarray:
        """
        Ensemble predictions from all learners.
        
        Combines T/S/X/R-Learner predictions via average.
        
        Returns:
            Ensemble CATE predictions
        """
        
        self.logger.info("Creating ensemble CATE...")
        
        cates = np.column_stack([
            self.models['TLearner']['cate'],
            self.models['SLearner']['cate'],
            self.models['XLearner']['cate'],
            self.models['RLearner']['cate']
        ])
        
        # Simple average
        ensemble_cate = cates.mean(axis=1)
        
        # Clip to reasonable range
        ensemble_cate = np.clip(ensemble_cate, -1, 1)
        
        self.cate_predictions['ensemble'] = ensemble_cate
        
        self.logger.info(f"Ensemble CATE: mean={ensemble_cate.mean():.4f}, std={ensemble_cate.std():.4f}")
        
        return ensemble_cate
    
    def identify_treatment_units(self, cate: np.ndarray, top_k: int = 500) -> Dict:
        """
        Identify customers with highest positive treatment effect.
        
        These are the best candidates for the service bundling offer.
        
        Args:
            cate: CATE predictions
            top_k: Number of top customers to identify
        
        Returns:
            Dictionary with customer indices and effects
        """
        
        # Sort by CATE (descending)
        indices = np.argsort(-cate)[:top_k]
        effects = cate[indices]
        
        result = {
            'customer_indices': indices,
            'treatment_effects': effects,
            'avg_effect': effects.mean(),
            'std_effect': effects.std(),
            'min_effect': effects.min(),
            'max_effect': effects.max(),
            'median_effect': np.median(effects)
        }
        
        self.logger.info(f"Top {top_k} customers identified for treatment")
        self.logger.info(f"  Average CATE: {result['avg_effect']:.4f}")
        self.logger.info(f"  Range: [{result['min_effect']:.4f}, {result['max_effect']:.4f}]")
        
        return result
    
    def compute_treatment_roi(
        self,
        treatment_recs: Dict,
        avg_monthly_revenue: float = 75.0,
        offer_cost: float = 20.0,
        months: int = 12
    ) -> Dict:
        """
        Compute ROI for treatment campaign.
        
        Args:
            treatment_recs: Output from identify_treatment_units()
            avg_monthly_revenue: Average monthly revenue per customer
            offer_cost: Cost of making bundling offer
            months: Campaign duration (months)
        
        Returns:
            Dictionary with ROI metrics
        """
        
        avg_effect = treatment_recs['avg_effect']
        num_customers = len(treatment_recs['customer_indices'])
        
        # Expected revenue saved per customer (avoided churn * revenue)
        potential_saved_per_customer = avg_effect * avg_monthly_revenue * months
        
        # ROI calculation
        cost_per_customer = offer_cost
        revenue_per_customer = potential_saved_per_customer - cost_per_customer
        
        roi_per_customer = (revenue_per_customer / cost_per_customer) if cost_per_customer > 0 else 0
        
        # Campaign totals
        total_cost = cost_per_customer * num_customers
        total_revenue = revenue_per_customer * num_customers
        campaign_roi = (total_revenue / total_cost) if total_cost > 0 else 0

        # Correctly calculate payback period
        monthly_revenue_saved = abs(avg_effect) * avg_monthly_revenue
        payback_period = (offer_cost / monthly_revenue_saved) if monthly_revenue_saved > 0 else float('inf')
        
        result = {
            'num_customers': num_customers,
            'avg_effect_pct': f"{avg_effect*100:.2f}%",
            'potential_saved_per_customer': f"${abs(potential_saved_per_customer):.2f}",
            'offer_cost': f"${cost_per_customer:.2f}",
            'roi_per_customer': f"{roi_per_customer*100:.2f}%",
            'total_cost': f"${total_cost:,.0f}",
            'total_revenue_saved': f"${abs(total_revenue):,.0f}",
            'campaign_roi': f"{campaign_roi*100:.2f}%",
            'payback_period_months': f"{payback_period:.2f}"
        }
        
        self.logger.info("ROI Analysis:")
        for key, value in result.items():
            self.logger.info(f"  {key}: {value}")
        
        return result
