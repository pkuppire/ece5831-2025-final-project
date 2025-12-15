# src/improved_cnn128.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import (
    ensure_dir, save_json, save_history, load_history,
    plot_history, evaluate_and_report, show_wrong_predictions
)
from data_eda import PlantDiseaseDataModule


@dataclass
class ImprovedCNNConfig:
    model_root: str = "models"
    experiment_name: str = "improved_cnn_ref"

    epochs: int = 50
    lr: float = 1e-3
    retrain: bool = False

    model_filename: str = "improved_cnn_ref.keras"

    # callbacks
    early_patience: int = 8
    use_reduce_lr: bool = True
    rlr_monitor: str = "val_loss"
    rlr_factor: float = 0.5
    rlr_patience: int = 2
    rlr_min_lr: float = 1e-6


class ImprovedCNNExperiment:
    """
    Conv2D(32) + BN + MaxPool
    Conv2D(64) + BN + MaxPool
    Conv2D(128) + BN + MaxPool
    Conv2D(256) + BN + MaxPool
    Flatten + Dense(256) + Dropout(0.5)
    Dense(num_classes, softmax)

    Compile: Adam(lr=1e-3), sparse_categorical_crossentropy, accuracy
    """
    def __init__(self, dm: PlantDiseaseDataModule, cfg: ImprovedCNNConfig):
        self.dm = dm
        self.cfg = cfg

        self.exp_dir = Path(cfg.model_root) / cfg.experiment_name
        self.model_path = self.exp_dir / cfg.model_filename
        self.history_path = self.exp_dir / "history.json"
        self.test_metrics_path = self.exp_dir / "test_metrics.json"
        self.meta_path = self.exp_dir / "meta.json"

        self.model: Optional[keras.Model] = None
        self.history: Optional[Dict] = None

    def build_model(self) -> keras.Model:
        input_shape = (*self.dm.cfg.img_size, 3)
        num_classes = len(self.dm.class_names)

        model = keras.Sequential(
            [
                layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2)),

                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2)),

                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2)),

                layers.Conv2D(256, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(pool_size=(2, 2)),

                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.5),

                layers.Dense(num_classes, activation="softmax"),
            ],
            name="improved_cnn_ref",
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.cfg.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _has_saved(self) -> bool:
        return self.model_path.exists() and self.history_path.exists()

    def load(self) -> None:
        self.model = keras.models.load_model(self.model_path)
        self.history = load_history(self.history_path)

    def train_or_load(self, train_ds, val_ds) -> None:
        ensure_dir(self.exp_dir)

        if (not self.cfg.retrain) and self._has_saved():
            print(f"[{self.cfg.experiment_name}] Loading saved model + history (no retrain).")
            self.load()
            return

        print(f"[{self.cfg.experiment_name}] Training...")
        self.model = self.build_model()

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_path),
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.cfg.early_patience,
                restore_best_weights=True,
                verbose=1,
            ),
        ]

        if self.cfg.use_reduce_lr:
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor=self.cfg.rlr_monitor,
                    factor=self.cfg.rlr_factor,
                    patience=self.cfg.rlr_patience,
                    min_lr=self.cfg.rlr_min_lr,
                    verbose=1,
                )
            )

        hist = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.cfg.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        self.history = hist.history
        save_history(self.history, self.history_path)

        save_json(
            {
                "experiment_name": self.cfg.experiment_name,
                "img_size": list(self.dm.cfg.img_size),
                "batch_size": self.dm.cfg.batch_size,
                "class_names": self.dm.class_names,
                "epochs": self.cfg.epochs,
                "lr": self.cfg.lr,
                "model_filename": self.cfg.model_filename,
                "arch": "Conv32/64/128/256 + BN + MaxPool + Flatten + Dense256 + Dropout0.5 + Softmax",
            },
            self.meta_path,
        )

      
        self.model = keras.models.load_model(self.model_path)

    def plot_curves(self) -> None:
        assert self.history is not None, "History missing; run train_or_load first."
        plot_history(
            self.history,
            title=f"{self.cfg.experiment_name} - Training Curves",
            save_path=self.exp_dir / "training_curves.png",
        )

    def evaluate_test(self, test_ds) -> Tuple[float, float]:
        assert self.model is not None, "Model missing; run train_or_load first."
        loss, acc, _, _ = evaluate_and_report(
            self.model, test_ds, class_names=self.dm.class_names, model_name=self.cfg.experiment_name
        )
        save_json({"test_loss": float(loss), "test_accuracy": float(acc)}, self.test_metrics_path)
        show_wrong_predictions(self.model, test_ds, self.dm.class_names, max_images=12)
        return float(loss), float(acc)
