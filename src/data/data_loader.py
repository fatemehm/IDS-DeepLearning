import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import yaml
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MQTTDataLoader:
    """Load and prepare MQTT IoT network traffic data"""
    
    def __init__(self, data_dir: str = "./data/raw", params_file: Optional[str] = None):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing CSV files
            params_file: Optional path to params.yaml for configuration
        """
        self.data_dir = Path(data_dir)
        self.params = self._load_params(params_file) if params_file else {}
        
        # File mappings - can be overridden by params.yaml
        self.file_mappings = self.params.get('data', {}).get('file_mappings', {
            'normal': 'biflow_normal.csv',
            'mqtt_bruteforce': 'biflow_mqtt_bruteforce.csv',
            'scan_aggressive': 'biflow_scan_A.csv',
            'scan_udp': 'biflow_scan_sU.csv',
            'sparta_ssh': 'biflow_sparta.csv'
        })
        
        # Label encoding for multi-class
        self.label_encoding = {
            'normal': 0,
            'mqtt_bruteforce': 1,
            'scan_aggressive': 2,
            'scan_udp': 3,
            'sparta_ssh': 4
        }
        
    def _load_params(self, params_file: str) -> Dict:
        """Load parameters from YAML file"""
        try:
            with open(params_file, 'r') as f:
                params = yaml.safe_load(f)
            logger.info(f"Loaded parameters from {params_file}")
            return params
        except Exception as e:
            logger.warning(f"Could not load params file: {e}")
            return {}
    
    def load_single_file(self, filename: str, attack_type: str) -> pd.DataFrame:
        """
        Load a single CSV file
        
        Args:
            filename: Name of the CSV file
            attack_type: Type of attack/traffic
            
        Returns:
            DataFrame with added attack_type and is_attack columns
        """
        filepath = self.data_dir / filename
        logger.info(f"Loading {filepath}")
        
        try:
            df = pd.read_csv(filepath, low_memory=False)
            
            # Add metadata columns
            df['attack_type'] = attack_type
            df['is_attack'] = 0 if attack_type == 'normal' else 1
            df['label_encoded'] = self.label_encoding[attack_type]
            
            logger.info(f"Loaded {len(df)} records from {filename}")
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            raise
    
    def load_all_data(self, 
                      balance_classes: bool = False,
                      sample_size: Optional[int] = None,
                      exclude_classes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load all CSV files and combine them
        
        Args:
            balance_classes: Whether to balance classes
            sample_size: Number of samples per class (if balancing)
            exclude_classes: List of attack types to exclude
            
        Returns:
            Combined DataFrame
        """
        dfs = []
        exclude_classes = exclude_classes or []
        
        for attack_type, filename in self.file_mappings.items():
            # Skip excluded classes
            if attack_type in exclude_classes:
                logger.info(f"Skipping {attack_type}")
                continue
                
            df = self.load_single_file(filename, attack_type)
            
            # Balance if requested
            if balance_classes and sample_size:
                if len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                    logger.info(f"Sampled {sample_size} records from {attack_type}")
            
            dfs.append(df)
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Shuffle the combined dataset
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Total records loaded: {len(combined_df)}")
        
        # Display class distribution
        self._log_class_distribution(combined_df)
        
        return combined_df
    
    def _log_class_distribution(self, df: pd.DataFrame):
        """Log the distribution of classes"""
        logger.info("\n" + "="*50)
        logger.info("CLASS DISTRIBUTION")
        logger.info("="*50)
        
        logger.info("\nBinary Classification (is_attack):")
        counts = df['is_attack'].value_counts()
        for label, count in counts.items():
            label_name = "Normal" if label == 0 else "Attack"
            percentage = (count / len(df)) * 100
            logger.info(f"  {label_name}: {count:,} ({percentage:.2f}%)")
        
        logger.info("\nMulti-class Classification (attack_type):")
        counts = df['attack_type'].value_counts()
        for attack, count in counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {attack}: {count:,} ({percentage:.2f}%)")
        
        logger.info("="*50 + "\n")
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding metadata)
        
        Args:
            df: DataFrame
            
        Returns:
            List of feature column names
        """
        metadata_cols = ['is_attack', 'attack_type', 'label_encoded']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        return feature_cols
    
    def create_binary_dataset(self, 
                             balance_classes: bool = False,
                             sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create binary classification dataset (Normal vs Attack)
        
        Args:
            balance_classes: Whether to balance classes
            sample_size: Number of samples per class
            
        Returns:
            X (features), y (binary labels)
        """
        df = self.load_all_data(balance_classes=balance_classes, sample_size=sample_size)
        
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols]
        y = df['is_attack']
        
        logger.info(f"\n📊 Binary dataset created:")
        logger.info(f"   Samples: {X.shape[0]:,}")
        logger.info(f"   Features: {X.shape[1]:,}")
        logger.info(f"   Class balance: {y.value_counts().to_dict()}\n")
        
        return X, y
    
    def create_multiclass_dataset(self,
                                  balance_classes: bool = False,
                                  sample_size: Optional[int] = None,
                                  use_encoded_labels: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create multi-class classification dataset (5 attack types)
        
        Args:
            balance_classes: Whether to balance classes
            sample_size: Number of samples per class
            use_encoded_labels: Use integer encoding (0-4) instead of string labels
            
        Returns:
            X (features), y (multi-class labels)
        """
        df = self.load_all_data(balance_classes=balance_classes, sample_size=sample_size)
        
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols]
        y = df['label_encoded'] if use_encoded_labels else df['attack_type']
        
        logger.info(f"\n📊 Multi-class dataset created:")
        logger.info(f"   Samples: {X.shape[0]:,}")
        logger.info(f"   Features: {X.shape[1]:,}")
        logger.info(f"   Classes: {y.nunique()}")
        logger.info(f"   Class distribution: {y.value_counts().to_dict()}\n")
        
        return X, y
    
    def get_data_info(self) -> Dict:
        """
        Get information about the available data
        
        Returns:
            Dictionary with data information
        """
        info = {
            'data_dir': str(self.data_dir),
            'files': {},
            'total_samples': 0
        }
        
        for attack_type, filename in self.file_mappings.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath, low_memory=False)
                info['files'][attack_type] = {
                    'filename': filename,
                    'samples': len(df),
                    'features': len(df.columns)
                }
                info['total_samples'] += len(df)
        
        return info
    
    def save_processed_data(self, 
                           df: pd.DataFrame, 
                           output_path: str,
                           compression: Optional[str] = 'gzip'):
        """
        Save processed data to file
        
        Args:
            df: DataFrame to save
            output_path: Output file path
            compression: Compression method ('gzip', 'bz2', 'zip', or None)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if compression:
            output_path = output_path.with_suffix(f'.csv.{compression}')
        
        df.to_csv(output_path, index=False, compression=compression)
        logger.info(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("\n" + "="*60)
    print("MQTT IDS Data Loader - Example Usage")
    print("="*60 + "\n")
    
    # Initialize loader
    loader = MQTTDataLoader(data_dir="./data/raw")
    
    # Get data info
    print("📁 Data Information:")
    info = loader.get_data_info()
    for attack_type, details in info['files'].items():
        print(f"   {attack_type}: {details['samples']:,} samples")
    print(f"   Total: {info['total_samples']:,} samples\n")
    
    # Example 1: Load binary classification data
    print("="*60)
    print("Example 1: Binary Classification Dataset")
    print("="*60)
    X_binary, y_binary = loader.create_binary_dataset()
    
    # Example 2: Load balanced multi-class data
    print("\n" + "="*60)
    print("Example 2: Balanced Multi-class Dataset")
    print("="*60)
    X_multi, y_multi = loader.create_multiclass_dataset(
        balance_classes=True, 
        sample_size=10000
    )

    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save multi-class dataset
    multiclass_file = output_dir / "multiclass_dataset.pkl"
    with open(output_dir / "multiclass_dataset.pkl", "wb") as f:
        pickle.dump((X_multi, y_multi), f)
    logger.info(f"✅ Multi-class dataset saved to {multiclass_file}")

    # Save binary dataset
    binary_file = output_dir / "binary_dataset.pkl"
    with open(output_dir / "binary_dataset.pkl", "wb") as f:
        pickle.dump((X_binary, y_binary), f)
    logger.info(f"✅ Binary dataset saved to {binary_file}")

    print("\n✅ Data loading examples completed!")