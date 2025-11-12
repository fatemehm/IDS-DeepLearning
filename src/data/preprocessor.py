import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import logging
from typing import Tuple, List, Optional, Dict
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiFlowPreprocessor:
    """
    Preprocessing pipeline for BiFlow network traffic data
    Handles MQTT IoT IDS dataset specific transformations
    """
    
    def __init__(self, 
                 scaling_method: str = 'standard',
                 handle_inf: bool = True,
                 handle_missing: bool = True):
        """
        Initialize preprocessor
        
        Args:
            scaling_method: 'standard', 'minmax', or 'robust'
            handle_inf: Replace infinite values
            handle_missing: Handle missing values
        """
        self.scaling_method = scaling_method
        self.handle_inf = handle_inf
        self.handle_missing = handle_missing
        
        # Will be fitted during preprocessing
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.feature_stats = None
        
        # Column categories (will be detected from data)
        self.ip_columns = []
        self.port_columns = []
        self.timestamp_columns = []
        self.numeric_columns = []
        self.categorical_columns = []
        
        logger.info(f"BiFlowPreprocessor initialized with {scaling_method} scaling")
    
    def identify_column_types(self, df: pd.DataFrame):
        """
        Automatically identify column types based on names and values
        
        Args:
            df: Input DataFrame
        """
        logger.info("Identifying column types...")
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Skip target columns
            if col in ['is_attack', 'attack_type', 'label_encoded']:
                continue
            
            # IP address columns (hex format like 0x...)
            if 'ip' in col_lower and df[col].dtype == 'object':
                self.ip_columns.append(col)
            
            # Port columns
            elif 'port' in col_lower:
                self.port_columns.append(col)
            
            # Timestamp columns
            elif 'time' in col_lower or col_lower.endswith('_at'):
                self.timestamp_columns.append(col)
            
            # Numeric columns
            elif pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_columns.append(col)
            
            # Categorical columns
            elif df[col].dtype == 'object':
                self.categorical_columns.append(col)
        
        logger.info(f"  IP columns: {len(self.ip_columns)}")
        logger.info(f"  Port columns: {len(self.port_columns)}")
        logger.info(f"  Timestamp columns: {len(self.timestamp_columns)}")
        logger.info(f"  Numeric columns: {len(self.numeric_columns)}")
        logger.info(f"  Categorical columns: {len(self.categorical_columns)}")
    
    def handle_ip_addresses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert IP addresses from hex to numeric features
        Options: hash, drop, or keep as categorical
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with processed IP columns
        """
        if not self.ip_columns:
            return df
        
        logger.info("Processing IP address columns...")
        df = df.copy()
        
        for col in self.ip_columns:
            try:
                # Option 1: Drop IP columns (they're identifiers, not features)
                df = df.drop(columns=[col])
                logger.info(f"  Dropped {col} (identifier)")
                
                # Option 2: Convert to numeric hash (uncomment if needed)
                # df[f'{col}_hash'] = df[col].apply(lambda x: hash(str(x)) % (10 ** 8))
                # df = df.drop(columns=[col])
                
            except Exception as e:
                logger.warning(f"  Error processing {col}: {e}")
        
        return df
    
    def handle_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process timestamp columns - extract useful features or drop
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with processed timestamp columns
        """
        if not self.timestamp_columns:
            return df
        
        logger.info("Processing timestamp columns...")
        df = df.copy()
        
        for col in self.timestamp_columns:
            try:
                # Option 1: Drop timestamps (absolute time not useful)
                df = df.drop(columns=[col])
                logger.info(f"  Dropped {col} (absolute time)")
                
                # Option 2: Extract time-based features (uncomment if needed)
                # df[col] = pd.to_datetime(df[col], errors='coerce')
                # df[f'{col}_hour'] = df[col].dt.hour
                # df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                # df = df.drop(columns=[col])
                
            except Exception as e:
                logger.warning(f"  Error processing {col}: {e}")
        
        return df
    
    def handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace infinite values with appropriate values
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with infinite values handled
        """
        if not self.handle_inf:
            return df
        
        logger.info("Handling infinite values...")
        df = df.copy()
        
        # Find columns with infinite values
        inf_cols = []
        for col in self.numeric_columns:
            if col in df.columns:
                if np.isinf(df[col]).any():
                    inf_cols.append(col)
        
        if inf_cols:
            logger.info(f"  Found infinite values in {len(inf_cols)} columns")
            
            for col in inf_cols:
                # Replace inf with NaN, then handle with median
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"  {col}: Replaced inf with median ({median_val:.2f})")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        if not self.handle_missing:
            return df
        
        logger.info("Handling missing values...")
        df = df.copy()
        
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        if len(missing_cols) > 0:
            logger.info(f"  Found missing values in {len(missing_cols)} columns")
            
            for col in missing_cols.index:
                missing_pct = (missing_cols[col] / len(df)) * 100
                
                if missing_pct > 50:
                    # Drop columns with >50% missing
                    df = df.drop(columns=[col])
                    logger.info(f"  {col}: Dropped ({missing_pct:.1f}% missing)")
                
                elif col in self.numeric_columns:
                    # Fill numeric with median
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(f"  {col}: Filled with median ({missing_pct:.1f}% missing)")
                
                else:
                    # Fill categorical with mode
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                    df[col] = df[col].fillna(mode_val)
                    logger.info(f"  {col}: Filled with mode ({missing_pct:.1f}% missing)")
        
        return df
    
    def handle_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        if not self.categorical_columns:
            return df
        
        logger.info("Encoding categorical features...")
        df = df.copy()
        
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
            
            unique_count = df[col].nunique()
            
            if unique_count > 50:
                # Too many categories - drop or hash
                df = df.drop(columns=[col])
                logger.info(f"  {col}: Dropped (too many categories: {unique_count})")
            
            elif unique_count == 2:
                # Binary encoding
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                logger.info(f"  {col}: Binary encoded")
            
            else:
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
                logger.info(f"  {col}: One-hot encoded ({unique_count} categories)")
        
        return df
    
    def remove_zero_variance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove features with zero or near-zero variance
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with zero-variance features removed
        """
        logger.info("Removing zero-variance features...")
        df = df.copy()
        
        # Calculate variance for numeric columns
        variances = df.var(numeric_only=True)
        zero_var_cols = variances[variances < 1e-10].index.tolist()
        
        if zero_var_cols:
            df = df.drop(columns=zero_var_cols)
            logger.info(f"  Removed {len(zero_var_cols)} zero-variance features")
        
        return df
    
    def create_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features from existing ones
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional engineered features
        """
        logger.info("Creating additional features...")
        df = df.copy()
        
        # Network flow ratios and rates
        if 'fwd_pkts_tot' in df.columns and 'bwd_pkts_tot' in df.columns:
            total_pkts = df['fwd_pkts_tot'] + df['bwd_pkts_tot']
            df['fwd_bwd_pkt_ratio'] = df['fwd_pkts_tot'] / (df['bwd_pkts_tot'] + 1)
            df['pkt_balance'] = (df['fwd_pkts_tot'] - df['bwd_pkts_tot']) / (total_pkts + 1)
            logger.info("  Created packet ratio features")
        
        if 'fwd_data_pkts_tot' in df.columns and 'bwd_data_pkts_tot' in df.columns:
            df['fwd_bwd_data_ratio'] = df['fwd_data_pkts_tot'] / (df['bwd_data_pkts_tot'] + 1)
            logger.info("  Created data packet ratio features")
        
        # Byte-based features
        if 'fwd_pkts_tot' in df.columns and 'fwd_data_pkts_tot' in df.columns:
            df['data_pkt_ratio'] = df['fwd_data_pkts_tot'] / (df['fwd_pkts_tot'] + 1)
        
        # IAT (Inter-Arrival Time) statistics
        iat_cols = [col for col in df.columns if 'iat' in col.lower()]
        if len(iat_cols) > 0:
            df['iat_variance'] = df[iat_cols].var(axis=1)
            df['iat_range'] = df[iat_cols].max(axis=1) - df[iat_cols].min(axis=1)
            logger.info("  Created IAT statistical features")
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numeric features
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for training, False for testing)
            
        Returns:
            Scaled DataFrame
        """
        logger.info(f"Scaling features using {self.scaling_method}...")
        df = df.copy()
        
        # Get numeric columns that exist in df
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_cols) == 0:
            logger.warning("  No numeric columns to scale")
            return df
        
        if fit:
            # Initialize scaler
            if self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling_method}")
            
            # Fit and transform
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            logger.info(f"  Fitted and transformed {len(numeric_cols)} features")
        
        else:
            # Transform only (using pre-fitted scaler)
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
            logger.info(f"  Transformed {len(numeric_cols)} features")
        
        return df
    
    def fit_transform(self, 
                     X: pd.DataFrame, 
                     y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit preprocessor and transform data (for training set)
        
        Args:
            X: Feature DataFrame
            y: Target Series (optional)
            
        Returns:
            Transformed DataFrame
        """
        logger.info("\n" + "="*70)
        logger.info("FITTING AND TRANSFORMING DATA (TRAINING SET)")
        logger.info("="*70 + "\n")
        
        df = X.copy()
        initial_shape = df.shape
        
        # 1. Identify column types
        self.identify_column_types(df)
        
        # 2. Handle IP addresses
        df = self.handle_ip_addresses(df)
        
        # 3. Handle timestamps
        df = self.handle_timestamps(df)
        
        # 4. Handle infinite values
        df = self.handle_infinite_values(df)
        
        # 5. Handle missing values
        df = self.handle_missing_values(df)
        
        # 6. Handle categorical features
        df = self.handle_categorical_features(df)
        
        # 7. Remove zero variance features
        df = self.remove_zero_variance_features(df)
        
        # 8. Create additional features
        df = self.create_additional_features(df)
        
        # 9. Scale features
        df = self.scale_features(df, fit=True)
        
        # Store feature names
        self.feature_names = df.columns.tolist()
        
        # Log transformation summary
        logger.info("\n" + "="*70)
        logger.info("TRANSFORMATION SUMMARY")
        logger.info("="*70)
        logger.info(f"  Input shape:  {initial_shape}")
        logger.info(f"  Output shape: {df.shape}")
        logger.info(f"  Features removed: {initial_shape[1] - df.shape[1]}")
        logger.info(f"  Features added: {df.shape[1] - initial_shape[1] + (initial_shape[1] - df.shape[1])}")
        logger.info("="*70 + "\n")
        
        return df
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor (for test/validation set)
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Transformed DataFrame
        """
        logger.info("\n" + "="*70)
        logger.info("TRANSFORMING DATA (TEST/VALIDATION SET)")
        logger.info("="*70 + "\n")
        
        if self.feature_names is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform() first.")
        
        df = X.copy()
        initial_shape = df.shape
        
        # Apply same transformations (without fitting)
        df = self.handle_ip_addresses(df)
        df = self.handle_timestamps(df)
        df = self.handle_infinite_values(df)
        df = self.handle_missing_values(df)
        df = self.handle_categorical_features(df)
        df = self.create_additional_features(df)
        
        # Ensure same columns as training
        missing_cols = set(self.feature_names) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        
        # Remove extra columns
        extra_cols = set(df.columns) - set(self.feature_names)
        df = df.drop(columns=list(extra_cols))
        
        # Reorder columns to match training
        df = df[self.feature_names]
        
        # Scale features
        df = self.scale_features(df, fit=False)
        
        logger.info(f"  Input shape:  {initial_shape}")
        logger.info(f"  Output shape: {df.shape}")
        logger.info("="*70 + "\n")
        
        return df
    
    def save(self, filepath: str):
        """
        Save fitted preprocessor to disk
        
        Args:
            filepath: Path to save the preprocessor
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'BiFlowPreprocessor':
        """
        Load fitted preprocessor from disk
        
        Args:
            filepath: Path to the saved preprocessor
            
        Returns:
            Loaded preprocessor
        """
        preprocessor = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor


if __name__ == "__main__":
    # Example usage
    print("\n" + "="*70)
    print("BiFlow Preprocessor - Example Usage")
    print("="*70 + "\n")
    
    # Load sample data
    from src.data.data_loader import MQTTDataLoader
    
    loader = MQTTDataLoader(data_dir="./data/raw")
    X, y = loader.create_binary_dataset()
    
    print(f"Original data shape: {X.shape}")
    
    # Initialize preprocessor
    preprocessor = BiFlowPreprocessor(
        scaling_method='standard',
        handle_inf=True,
        handle_missing=True
    )
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    
    print(f"Processed data shape: {X_processed.shape}")
    print(f"Feature names: {X_processed.columns.tolist()[:10]}...")
    
    # Save preprocessor
    preprocessor.save('./models/preprocessor.pkl')
    
    processed_path = Path("data/processed/preprocessed_multiclass.pkl")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(processed_path, "wb") as f:
        pickle.dump((X_processed, y), f)  # save both features and labels
    logger.info(f"Processed dataset saved to {processed_path}")
    
    print("\n✅ Preprocessing example completed!")