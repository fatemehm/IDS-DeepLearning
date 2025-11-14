"""
Professional Data Loader with Validation
Author: Mahshid Zadeh
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from dataclasses import dataclass
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Dataset metadata"""

    name: str
    total_samples: int
    num_features: int
    attack_distribution: Dict[str, int]
    data_hash: str


class MQTTDataLoader:
    """
    Professional data loader for MQTT IoT network traffic

    Features:
    - Data validation
    - Automatic feature detection
    - Class balancing
    - Train/val/test splitting
    - Data quality checks
    """

    def __init__(self, config: Dict):
        """
        Initialize data loader

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.data_dir = Path(config["data"]["raw_dir"])
        self.attack_types = config["data"]["attack_types"]

        # File mappings
        self.file_mappings = config["data"]["file_mappings"]

        # Label encoding
        self.label_encoding = {attack: idx for idx, attack in enumerate(self.attack_types)}

        logger.info("DataLoader initialized")
        logger.info("   Attack types: {len(self.attack_types)}")
        logger.info("   Data directory: {self.data_dir}")

    def compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash for data versioning and integrity check"""
        data_str = pd.util.hash_pandas_object(df).values.tobytes()
        return hashlib.md5(data_str).hexdigest()[:8]

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data quality

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check for nulls
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            issues.append(f"Found {null_count} null values")

        # Check for duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            issues.append(f"Found {dup_count} duplicate rows")

        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        if inf_count > 0:
            issues.append(f"Found {inf_count} infinite values")

        is_valid = len(issues) == 0

        if is_valid:
            logger.info("Data validation passed")
        else:
            logger.warning(f"Data validation issues: {issues}")

        return is_valid, issues

    def load_single_file(
        self, filename: str, attack_type: str, sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load single CSV file

        Args:
            filename: CSV filename
            attack_type: Type of attack/traffic
            sample_size: Optional limit on number of samples

        Returns:
            DataFrame with metadata columns added
        """
        filepath = self.data_dir / filename

        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"Data file not found: {filepath}")

        logger.info("Loading {filename}...")

        try:
            # Load CSV
            df = pd.read_csv(filepath, low_memory=False)
            original_size = len(df)

            # Sample if requested
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                logger.info("   Sampled {sample_size:,} from {original_size:,} records")

            # Add metadata columns
            df["attack_type"] = attack_type
            df["is_attack"] = 0 if attack_type == "normal" else 1
            df["label_encoded"] = self.label_encoding[attack_type]

            logger.info("Loaded {len(df):,} records ({attack_type})")

            return df

        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            raise

    def load_all_data(
        self, balance_classes: bool = True, sample_per_class: Optional[int] = None
    ) -> Tuple[pd.DataFrame, DatasetInfo]:
        """
        Load all dataset files

        Args:
            balance_classes: Whether to balance class sizes
            sample_per_class: Number of samples per class if balancing

        Returns:
            (combined_dataframe, dataset_info)
        """
        logger.info("\n" + "=" * 70)
        logger.info("LOADING DATASET")
        logger.info("=" * 70)

        dfs = []

        # Load each attack type
        for attack_type, filename in self.file_mappings.items():
            try:
                df = self.load_single_file(
                    filename, attack_type, sample_size=sample_per_class if balance_classes else None
                )
                dfs.append(df)
            except FileNotFoundError:
                logger.warning(f"Skipping {attack_type} - file not found")
                continue

        if not dfs:
            raise ValueError("No data files could be loaded!")

        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)

        # Shuffle
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info("\nDataset Statistics:")
        logger.info("   Total samples: {len(combined_df):,}")
        logger.info("   Total features: {len(combined_df.columns) - 3}")

        # Validate data
        is_valid, issues = self.validate_data(combined_df)

        # Class distribution
        logger.info("\nClass Distribution:")
        for attack_type in combined_df["attack_type"].value_counts().index:
            count = len(combined_df[combined_df["attack_type"] == attack_type])
            percentage = (count / len(combined_df)) * 100
            logger.info("   {attack_type}: {count:,} ({percentage:.1f}%)")

        # Create dataset info
        info = DatasetInfo(
            name="iot_ids_dataset",
            total_samples=len(combined_df),
            num_features=len(combined_df.columns) - 3,
            attack_distribution=combined_df["attack_type"].value_counts().to_dict(),
            data_hash=self.compute_data_hash(combined_df),
        )

        logger.info("\nData hash: {info.data_hash}")
        logger.info("=" * 70 + "\n")

        return combined_df, info

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding metadata)"""
        metadata_cols = ["is_attack", "attack_type", "label_encoded"]
        return [col for col in df.columns if col not in metadata_cols]

    def create_splits(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, val_size: float = 0.2
    ) -> Dict[str, Tuple]:
        """
        Create train/val/test splits with stratification

        Args:
            X: Features
            y: Labels
            test_size: Test set proportion
            val_size: Validation set proportion

        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
        )

        splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}

        logger.info("Dataset splits created:")
        for name, (X_split, y_split) in splits.items():
            logger.info("   {name:5s}: {len(X_split):,} samples")

        return splits


if __name__ == "__main__":
    # Test the data loader
    import sys

    sys.path.append(".")
    from src.config import config

    print("\nTesting Data Loader...")

    loader = MQTTDataLoader(config.config)

    # Load data
    df, info = loader.load_all_data(balance_classes=True, sample_per_class=10000)

    print(f"\nTest successful!")
    print(f"   Samples: {info.total_samples:,}")
    print(f"   Features: {info.num_features}")
    print(f"   Hash: {info.data_hash}")
