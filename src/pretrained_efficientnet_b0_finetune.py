from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import (
    ensure_dir, save_json, save_history, load_history,
    plot_history, evaluate_and_report, show_wrong_predictions
)
from data_eda import PlantDiseaseDataModule


@dataclass
class EfficientNetB0SafeFTConfig:
    model_root: str = "models"

   
    experiment_name: str = "efficientnetb0_ft_safe_224_last20_lr1e5"
    model_filename: str = "efficientnetb0_ft_safe_224_last20_lr1e5.keras"

    
    base_experiment_name: str = "efficientnetb0_tl_224"
    base_model_filename: str = "efficientnetb0_tl_224.keras"

    epochs: int = 10
    lr: float = 1e-5
    retrain: bool = True

    unfreeze_last_n: int = 20
    keep_batchnorm_frozen: bool = True

    early_patience: int = 3
    use_reduce_lr: bool = True
    rlr_monitor: str = "val_loss"
    rlr_factor: float = 0.5
    rlr_patience: int = 1
    rlr_min_lr: float = 1e-7


class EfficientNetB0SafeFineTuneExperiment:
    

    def __init__(self, dm: PlantDiseaseDataModule, cfg: EfficientNetB0SafeFTConfig):
        self.dm = dm
        self.cfg = cfg

        self.exp_dir = Path(cfg.model_root) / cfg.experiment_name
        self.model_path = self.exp_dir / cfg.model_filename
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

    def _list_available_models(self) -> str:
        # list all keras files under models/**/
        pattern = str(Path(self.cfg.model_root) / "**" / "*.keras")
        found = glob.glob(pattern, recursive=True)
        found = sorted(found)
        if not found:
            return "No .keras files found under models/."
        return "\n".join(found)

    def _load_base_model(self) -> keras.Model:
        base_path = Path(self.cfg.model_root) / self.cfg.base_experiment_name / self.cfg.base_model_filename
        print(f"[{self.cfg.experiment_name}] Base model path: {base_path}")

        if not base_path.exists():
            available = self._list_available_models()
            raise FileNotFoundError(
                f"Base TL model not found:\n  {base_path}\n\n"
                f"Available saved models under '{self.cfg.model_root}':\n{available}\n\n"
                f"Fix your config to point to an existing TL model, e.g.\n"
                f'  base_experiment_name="efficientnetb0_tl_224"\n'
                f'  base_model_filename="efficientnetb0_tl_224.keras"\n'
            )
        return keras.models.load_model(base_path)

    def _make_callbacks(self):
        cbs = [
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_path),
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
            cbs.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor=self.cfg.rlr_monitor,
                    factor=self.cfg.rlr_factor,
                    patience=self.cfg.rlr_patience,
                    min_lr=self.cfg.rlr_min_lr,
                    verbose=1,
                )
            )
        return cbs

    def _set_trainable_safe(self):
       
        assert self.model is not None

        for l in self.model.layers:
            l.trainable = False

        head_types = (layers.GlobalAveragePooling2D, layers.Dropout, layers.Dense)
        for l in self.model.layers:
            if isinstance(l, head_types):
                l.trainable = True

        non_head_layers = [l for l in self.model.layers if not isinstance(l, head_types)]
        n = int(self.cfg.unfreeze_last_n)
        if n > 0 and len(non_head_layers) > 0:
            for l in non_head_layers[-n:]:
                l.trainable = True

        if self.cfg.keep_batchnorm_frozen:
            for l in self.model.layers:
                if isinstance(l, layers.BatchNormalization):
                    l.trainable = False

        trainable_count = sum(int(l.trainable) for l in self.model.layers)
        print(f"[{self.cfg.experiment_name}] Trainable layers after setup: {trainable_count}/{len(self.model.layers)}")

    def train_or_load(self, train_ds, val_ds) -> None:
        ensure_dir(self.exp_dir)

        if (not self.cfg.retrain) and self._has_saved():
            print(f"[{self.cfg.experiment_name}] Loading saved model + history (no retrain).")
            self.load()
            return

        print(
            f"[{self.cfg.experiment_name}] Fine-tuning from '{self.cfg.base_experiment_name}' "
            f"(unfreeze_last_n={self.cfg.unfreeze_last_n}, lr={self.cfg.lr})"
        )

        self.model = self._load_base_model()
        self._set_trainable_safe()

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.cfg.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        hist = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.cfg.epochs,
            callbacks=self._make_callbacks(),
            verbose=1,
        )

        self.history = hist.history
        save_history(self.history, self.history_path)

        save_json({
            "experiment_name": self.cfg.experiment_name,
            "base_experiment_name": self.cfg.base_experiment_name,
            "img_size": list(self.dm.cfg.img_size),
            "batch_size": self.dm.cfg.batch_size,
            "class_names": self.dm.class_names,
            "epochs": self.cfg.epochs,
            "lr": self.cfg.lr,
            "unfreeze_last_n": self.cfg.unfreeze_last_n,
            "keep_batchnorm_frozen": self.cfg.keep_batchnorm_frozen,
            "fine_tuning": True,
            "checkpoint_monitor": "val_loss"
        }, self.meta_path)

        self.model = keras.models.load_model(self.model_path)

    def plot_curves(self) -> None:
        assert self.history is not None
        plot_history(
            self.history,
            title=f"{self.cfg.experiment_name} - Fine-tuning Curves",
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
