"""
MLflow Experiment Tracking with DagsHub
Author: fatemehm
"""
import mlflow
import mlflow.tensorflow
from typing import Dict, Any, Optional
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentTracker:
    """MLflow experiment tracking wrapper with DagsHub support"""

    def __init__(self, tracking_uri: str, experiment_name: str):
        """
        Initialize tracker

        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of experiment
        """
        # Set credentials from environment
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.experiment_name = experiment_name
        self.active_run = None

        logger.info(f"✅ MLflow Tracker initialized")
        logger.info(f"   Tracking URI: {tracking_uri}")
        logger.info(f"   Experiment: {experiment_name}")

    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """Start new MLflow run"""
        self.active_run = mlflow.start_run(run_name=run_name, tags=tags or {})
        logger.info(f"🚀 Started run: {run_name}")
        return self.active_run

    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)
        logger.info(f"📝 Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, artifact_path: str = "model"):
        """Log TensorFlow model"""
        mlflow.tensorflow.log_model(model, artifact_path)
        logger.info(f"💾 Model logged to MLflow")

    def log_artifact(self, local_path: str):
        """Log artifact file"""
        mlflow.log_artifact(local_path)

    def end_run(self):
        """End current run"""
        if self.active_run:
            mlflow.end_run()
            logger.info(f"✅ Run completed")
            self.active_run = None


if __name__ == "__main__":
    # Test tracker
    tracker = ExperimentTracker(
        tracking_uri="https://dagshub.com/fatemehm/IDS-DeepLearning.mlflow", experiment_name="test"
    )

    try:
        with tracker.start_run("test_run"):
            tracker.log_params({"learning_rate": 0.001})
            tracker.log_metrics({"accuracy": 0.95})
        print("✅ Tracker test successful! DagsHub connection working!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure:")
        print("   1. You created .env file with your DagsHub token")
        print("   2. Your token has the correct permissions")
        print("   3. The repository exists on DagsHub")
