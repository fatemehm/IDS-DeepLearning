"""
Professional Data Preprocessor - Fixed for IP addresses
Author: Mahshid Zadeh
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, Union
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiFlowPreprocessor:
    """Preprocessing pipeline for BiFlow network traffic data"""

    def __init__(
        self, scaling_method: str = "standard", handle_inf: bool = True, handle_missing: bool = True
    ):
        self.scaling_method = scaling_method
        self.handle_inf = handle_inf
        self.handle_missing = handle_missing

        self.scaler = None
        self.feature_names = None

        logger.info("Preprocessor initialized (scaling: {scaling_method})")

    def _drop_non_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop non-numeric columns (IP addresses, strings, etc.)"""
        df = df.copy()

        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

        if non_numeric_cols:
            logger.info("   Dropping {len(non_numeric_cols)} non-numeric columns")
            logger.info("   Non-numeric columns: {non_numeric_cols[:5]}...")
            df = df[numeric_cols]

        return df

    def _handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite values"""
        if not self.handle_inf:
            return df

        df = df.copy()
        inf_cols = []

        for col in df.columns:
            if np.isinf(df[col]).any():
                inf_cols.append(col)

        if inf_cols:
            logger.info("   Handling {len(inf_cols)} columns with infinite values")
            for col in inf_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        if not self.handle_missing:
            return df

        df = df.copy()
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]

        if len(missing_cols) > 0:
            logger.info("   Handling {len(missing_cols)} columns with missing values")
            for col in missing_cols.index:
                missing_pct = (missing_cols[col] / len(df)) * 100
                if missing_pct > 50:
                    df = df.drop(columns=[col])
                    logger.info("     Dropped {col} ({missing_pct:.1f}% missing)")
                else:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)

        return df

    def _scale_features(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Scale features and return numpy array"""
        if fit:
            if self.scaling_method == "standard":
                self.scaler = StandardScaler()
            elif self.scaling_method == "minmax":
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling_method}")

            scaled = self.scaler.fit_transform(df)
            logger.info("   Scaled {df.shape[1]} features using {self.scaling_method}")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted")
            scaled = self.scaler.transform(df)

        return scaled

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform - returns numpy array"""
        logger.info("\nFitting and transforming data...")

        df = X.copy()
        initial_shape = df.shape

        # 1. Drop non-numeric columns (IP addresses, etc.)
        df = self._drop_non_numeric_columns(df)

        # 2. Handle infinite values
        df = self._handle_infinite_values(df)

        # 3. Handle missing values
        df = self._handle_missing_values(df)

        # 4. Scale and get numpy array
        scaled_array = self._scale_features(df, fit=True)

        # Store feature names for later
        self.feature_names = df.columns.tolist()

        logger.info("Preprocessing complete:")
        logger.info("   Input shape: {initial_shape}")
        logger.info("   Output shape: {scaled_array.shape}")
        logger.info("   Features: {len(self.feature_names)}\n")

        return scaled_array

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform - returns numpy array"""
        if self.feature_names is None:
            raise ValueError("Preprocessor not fitted")

        logger.info("Transforming data...")

        df = X.copy()

        # Keep only numeric columns
        df = self._drop_non_numeric_columns(df)

        # Apply transformations
        df = self._handle_infinite_values(df)
        df = self._handle_missing_values(df)

        # Ensure same columns as training
        missing_cols = set(self.feature_names) - set(df.columns)
        for col in missing_cols:
            df[col] = 0

        extra_cols = set(df.columns) - set(self.feature_names)
        if extra_cols:
            df = df.drop(columns=list(extra_cols))

        # Reorder to match training
        df = df[self.feature_names]

        # Scale and return numpy array
        scaled_array = self._scale_features(df, fit=False)

        logger.info("Transform complete: {scaled_array.shape}\n")

        return scaled_array

    def save(self, filepath: str):
        """Save preprocessor"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Preprocessor saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> "BiFlowPreprocessor":
        """Load preprocessor"""
        preprocessor = joblib.load(filepath)
        logger.info("Preprocessor loaded from {filepath}")
        return preprocessor


if __name__ == "__main__":
    # Test
    import sys

    sys.path.append(".")
    from src.data.data_loader import MQTTDataLoader
    from src.config import config

    print("\nTesting Preprocessor...")

    loader = MQTTDataLoader(config.config)
    df, _ = loader.load_all_data(balance_classes=True, sample_per_class=1000)

    feature_cols = loader.get_feature_columns(df)
    X = df[feature_cols]

    preprocessor = BiFlowPreprocessor(scaling_method="standard")
    X_processed = preprocessor.fit_transform(X)

    print(f"Test successful!")
    print(f"   Original shape: {X.shape}")
    print(f"   Processed shape: {X_processed.shape}")
    print(f"   Data type: {type(X_processed)}")
