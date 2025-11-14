"""Configuration management"""
import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration class with dot notation access"""

    config_path: Path
    config: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: str = "configs/config.yaml"):
        """Load configuration from YAML file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r") as f:
            config = yaml.safe_load(f)

        return cls(config_path=path, config=config)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation (e.g., 'data.raw_dir')"""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)

    def __repr__(self) -> str:
        return f"Config(path={self.config_path})"


# Global config instance
config = Config.from_yaml()


if __name__ == "__main__":
    # Test configuration
    print("Testing configuration...")
    print(f"Project name: {config.get('project.name')}")
    print(f"Data directory: {config.get('data.raw_dir')}")
    print(f"MLflow URI: {config.get('mlflow.tracking_uri')}")
    print("Configuration loaded successfully!")
