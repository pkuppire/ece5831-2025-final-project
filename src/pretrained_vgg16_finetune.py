# src/pretrained_vgg16_finetune.py
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
class VGG16FTConfig:
    model_root: str = "models"
    experiment_name: str = "vgg_finetuned_224"
    model_filename: str = "vgg_finetuned_224.keras"

 
    epochs_stage1: int = 10
    epochs_stage2: int = 8
    lr_stage1: float = 1e-3
    lr_stage2: float = 1e-4

  
    trainable_from: int = 11

    retrain: bool = True

    dense_units: int = 256
    dropout: float = 0.5

   
    early_patience: int = 5
    use_reduce_lr: bool = True
    rlr_monitor: str = "val_loss"
    rlr_factor: float = 0.5
    rlr_patience: int = 2
    rlr_min_lr: float = 1e-6


class VGG16FineTuneExperiment:
    
    def __init__(self, dm: PlantDiseaseDataModule, cfg: VGG16FTConfig):
        self.dm = dm
        self.cfg = cfg

        self.exp_dir = Path(cfg.model_root) / cfg.experiment_name
        self.model_path = self.exp_dir / cfg.model_filename
        self.stage1_path = self.exp_dir / f"{cfg.experiment_name}_stage1_best.keras"
        self.history_path = self.exp_dir / "history.json"
        self.test_metrics_path = self.exp_dir / "test_metrics.json"
        self.meta_path = self.exp_dir / "meta.json"

        self.model: Optional[keras.Model] = None
        self.history: Optional[Dict] = None

    def _has_saved(self) -> bool:
        return self.model_path.exists() and self.history_path.exists()

    def load(self) -> None:
        self.model = keras.models.load_model(self.model_path)
        self.history = load_history(self.history_path)

    def build_model(self, trainable_from: Optional[int]) -> keras.Model:
        input_shape = (*self.dm.cfg.img_size, 3)
        num_classes = len(self.dm.class_names)

        inputs = keras.Input(shape=input_shape, name="image")

        x = layers.Rescaling(255.0, name="to_255")(inputs)
        x = tf.keras.applications.vgg16.preprocess_input(x)

        backbone = tf.keras.applications.VGG16(
            include_top=False,
            weights="imagenet",
            input_tensor=x,
            pooling=None
        )

        if trainable_from is None:
            backbone.trainable = False
        else:
            backbone.trainable = True
            for i, layer in enumerate(backbone.layers):
                layer.trainable = (i >= int(trainable_from))

        x = backbone.output
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.Dense(self.cfg.dense_units, activation="relu", name="dense256")(x)
        x = layers.BatchNormalization(name="bn")(x)
        x = layers.Dropout(self.cfg.dropout, name="drop")(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

        model = keras.Model(inputs, outputs, name=self.cfg.experiment_name)
        return model

    def _make_callbacks(self, ckpt_path: Path):
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=str(ckpt_path),
                monitor="val_loss",
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
        return callbacks

    def train_or_load(self, train_ds, val_ds) -> None:
        ensure_dir(self.exp_dir)

        if (not self.cfg.retrain) and self._has_saved():
            print(f"[{self.cfg.experiment_name}] Loading saved model + history (no retrain).")
            self.load()
            return

       
        print(f"[{self.cfg.experiment_name}] Stage 1: frozen backbone (epochs={self.cfg.epochs_stage1}, lr={self.cfg.lr_stage1})")
        stage1 = self.build_model(trainable_from=None)
        stage1.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.cfg.lr_stage1),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        h1 = stage1.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.cfg.epochs_stage1,
            callbacks=self._make_callbacks(self.stage1_path),
            verbose=1,
        )

        best_stage1 = keras.models.load_model(self.stage1_path)

        print(f"[{self.cfg.experiment_name}] Stage 2: unfreeze from layer {self.cfg.trainable_from} (epochs={self.cfg.epochs_stage2}, lr={self.cfg.lr_stage2})")
        stage2 = self.build_model(trainable_from=self.cfg.trainable_from)
        stage2.set_weights(best_stage1.get_weights())

        stage2.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.cfg.lr_stage2),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        h2 = stage2.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.cfg.epochs_stage2,
            callbacks=self._make_callbacks(self.model_path),
            verbose=1,
        )

        hist1 = h1.history
        hist2 = h2.history
        merged = {}
        for k in set(list(hist1.keys()) + list(hist2.keys())):
            merged[k] = list(hist1.get(k, [])) + list(hist2.get(k, []))
        self.history = merged
        save_history(self.history, self.history_path)

        save_json({
            "experiment_name": self.cfg.experiment_name,
            "img_size": list(self.dm.cfg.img_size),
            "batch_size": self.dm.cfg.batch_size,
            "class_names": self.dm.class_names,
            "epochs_stage1": self.cfg.epochs_stage1,
            "epochs_stage2": self.cfg.epochs_stage2,
            "lr_stage1": self.cfg.lr_stage1,
            "lr_stage2": self.cfg.lr_stage2,
            "backbone": "VGG16(ImageNet)",
            "fine_tuning": True,
            "trainable_from": self.cfg.trainable_from,
            "checkpoint_monitor": "val_loss"
        }, self.meta_path)

        self.model = keras.models.load_model(self.model_path)

    def plot_curves(self) -> None:
        assert self.history is not None
        plot_history(
            self.history,
            title=f"{self.cfg.experiment_name} - Training Curves (Stage1+Stage2)",
            save_path=self.exp_dir / "training_curves.png"
        )

    def evaluate_test(self, test_ds) -> Tuple[float, float]:
        assert self.model is not None
        loss, acc, _, _ = evaluate_and_report(
            self.model, test_ds,
            class_names=self.dm.class_names,
            model_name=self.cfg.experiment_name
        )
        save_json({"test_loss": float(loss), "test_accuracy": float(acc)}, self.test_metrics_path)
        show_wrong_predictions(self.model, test_ds, self.dm.class_names, max_images=12)
        return float(loss), float(acc)
