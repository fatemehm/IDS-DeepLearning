"""
Model Training Pipeline with MLflow Tracking
Author: Mahshid Zadeh
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime
import logging
import sys
from pathlib import Path
import json
from typing import Dict, Optional

sys.path.append(".")
from src.config import config
from src.data.data_loader import MQTTDataLoader
from src.data.preprocessor import BiFlowPreprocessor
from src.training.tracker import ExperimentTracker
import mlflow

mlflow.set_tracking_uri("mlruns")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelBuilder:
    """Build different model architectures"""

    @staticmethod
    def build_dnn(input_shape: int, num_classes: int, config: dict) -> keras.Model:
        """Build Deep Neural Network"""
        dnn_config = config["model"]["dnn"]

        model = keras.Sequential(name="dnn")
        model.add(keras.layers.Input(shape=(input_shape,)))

        # Hidden layers
        for units in dnn_config["hidden_layers"]:
            model.add(
                keras.layers.Dense(
                    units,
                    activation=dnn_config["activation"],
                    kernel_regularizer=keras.regularizers.l2(dnn_config["l2_regularization"]),
                )
            )
            if dnn_config["use_batch_norm"]:
                model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(dnn_config["dropout_rate"]))

        # Output layer
        model.add(keras.layers.Dense(num_classes, activation="softmax"))

        return model

    @staticmethod
    def build_lstm(input_shape: int, num_classes: int, config: Dict) -> keras.Model:
        """Build LSTM model"""
        lstm_config = config["model"]["lstm"]

        model = keras.Sequential(name="lstm")
        model.add(keras.layers.Input(shape=(1, input_shape)))

        for i, units in enumerate(lstm_config["units"]):
            return_sequences = i < len(lstm_config["units"]) - 1
            model.add(
                keras.layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=lstm_config["dropout_rate"],
                    recurrent_dropout=lstm_config["recurrent_dropout"],
                )
            )

        model.add(keras.layers.Dense(num_classes, activation="softmax"))

        return model

    @staticmethod
    def build_gru(input_shape: int, num_classes: int, config: Dict) -> keras.Model:
        """Build GRU model"""
        gru_config = config["model"]["gru"]

        model = keras.Sequential(name="gru")
        model.add(keras.layers.Input(shape=(1, input_shape)))

        for i, units in enumerate(gru_config["units"]):
            return_sequences = i < len(gru_config["units"]) - 1
            model.add(
                keras.layers.GRU(
                    units,
                    return_sequences=return_sequences,
                    dropout=gru_config["dropout_rate"],
                    recurrent_dropout=gru_config["recurrent_dropout"],
                )
            )

        model.add(keras.layers.Dense(num_classes, activation="softmax"))

        return model


class Trainer:
    """Model training pipeline"""

    def __init__(self, config_dict: Dict):
        self.config = config_dict
        self.model = None
        self.history = None
        self.preprocessor = None

        # Initialize tracker
        self.tracker = ExperimentTracker(
            tracking_uri=config_dict["mlflow"]["tracking_uri"],
            experiment_name=config_dict["mlflow"]["experiment_name"],
        )

    def prepare_data(self):
        """Load and prepare data"""
        logger.info("\n" + "=" * 70)
        logger.info("PREPARING DATA")
        logger.info("=" * 70)

        # Load data
        loader = MQTTDataLoader(self.config)
        df, info = loader.load_all_data(
            balance_classes=self.config["data"]["preprocessing"]["balance_classes"],
            sample_per_class=self.config["data"]["preprocessing"]["sample_per_class"],
        )

        # Separate features and labels
        feature_cols = loader.get_feature_columns(df)
        X = df[feature_cols]
        y = df["label_encoded"]

        # Create splits
        splits = loader.create_splits(
            X,
            y,
            test_size=self.config["data"]["preprocessing"]["test_size"],
            val_size=self.config["data"]["preprocessing"]["val_size"],
        )

        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_test, y_test = splits["test"]

        # Preprocess
        logger.info("\n" + "=" * 70)
        logger.info("PREPROCESSING")
        logger.info("=" * 70)

        self.preprocessor = BiFlowPreprocessor(
            scaling_method=self.config["data"]["preprocessing"]["scaling_method"]
        )

        X_train = self.preprocessor.fit_transform(X_train)
        X_val = self.preprocessor.transform(X_val)
        X_test = self.preprocessor.transform(X_test)

        # Convert to numpy and one-hot encode labels
        # X_train = X_train.values
        # X_val = X_val.values
        # X_test = X_test.values

        y_train = keras.utils.to_categorical(y_train, num_classes=5)
        y_val = keras.utils.to_categorical(y_val, num_classes=5)
        y_test = keras.utils.to_categorical(y_test, num_classes=5)

        logger.info(f"\nData prepared:")
        logger.info(f"   Train: {X_train.shape}")
        logger.info(f"   Val:   {X_val.shape}")
        logger.info(f"   Test:  {X_test.shape}\n")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test), info

    def train(self, model_type: str = "dnn", run_name: Optional[str] = None):
        """Train model with MLflow tracking"""

        run_name = run_name or f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info("\n" + "=" * 70)
        logger.info(f"TRAINING {model_type.upper()} MODEL")
        logger.info("=" * 70)

        # Prepare data
        (X_train, y_train), (X_val, y_val), (X_test, y_test), data_info = self.prepare_data()

        # Start MLflow run
        with self.tracker.start_run(run_name, tags={"model": model_type}):
            # Build model
            builder = ModelBuilder()
            if model_type == "dnn":
                self.model = builder.build_dnn(X_train.shape[1], 5, self.config)
            elif model_type == "lstm":
                X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
                X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
                self.model = builder.build_lstm(X_train.shape[2], 5, self.config)
            elif model_type == "gru":
                X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
                X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
                self.model = builder.build_gru(X_train.shape[2], 5, self.config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Log parameters
            train_config = self.config["model"]["training"]
            params = {
                "model_type": model_type,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "num_features": X_train.shape[-1],
                "data_hash": data_info.data_hash,
                **train_config,
            }
            self.tracker.log_params(params)

            # Compile
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=train_config["learning_rate"]),
                loss="categorical_crossentropy",
                metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
            )

            logger.info("\nModel Architecture:")
            self.model.summary()

            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=train_config["early_stopping_patience"],
                    restore_best_weights=True,
                    verbose=1,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=train_config["reduce_lr_patience"],
                    min_lr=1e-7,
                    verbose=1,
                ),
                keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: self.tracker.log_metrics(logs, step=epoch)
                ),
            ]

            # Train
            logger.info("\nStarting training...")
            self.history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=train_config["epochs"],
                batch_size=train_config["batch_size"],
                callbacks=callbacks,
                verbose=1,
            )

            # Evaluate on test set
            logger.info("\nEvaluating on test set...")
            test_results = self.model.evaluate(X_test, y_test, verbose=0)

            final_metrics = {
                "test_loss": float(test_results[0]),
                "test_accuracy": float(test_results[1]),
                "test_precision": float(test_results[2]),
                "test_recall": float(test_results[3]),
                "best_epoch": int(np.argmax(self.history.history["val_accuracy"])),
            }

            self.tracker.log_metrics(final_metrics)

            # Save model
            model_path = Path("models/trained") / f"{run_name}.h5"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(model_path)

            # Save preprocessor
            preprocessor_path = Path("models/trained") / f"{run_name}_preprocessor.pkl"
            self.preprocessor.save(str(preprocessor_path))

            # Log to MLflow
            self.tracker.log_model(self.model)
            self.tracker.log_artifact(str(preprocessor_path))

            logger.info("\n" + "=" * 70)
            logger.info("TRAINING COMPLETE!")
            logger.info("=" * 70)
            logger.info(f"Test Accuracy:  {final_metrics['test_accuracy']:.4f}")
            logger.info(f"Test Precision: {final_metrics['test_precision']:.4f}")
            logger.info(f"Test Recall:    {final_metrics['test_recall']:.4f}")
            logger.info(f"Model saved:    {model_path}")
            logger.info("=" * 70 + "\n")

            return final_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train IoT IDS model")
    parser.add_argument("--model", type=str, default="dnn", choices=["dnn", "lstm", "gru"])
    parser.add_argument("--name", type=str, default=None, help="Run name")
    args = parser.parse_args()

    trainer = Trainer(config.config)
    trainer.train(model_type=args.model, run_name=args.name)
