# src/data_eda.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import hashlib
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class DataConfig:
    data_root: str
    img_size: Tuple[int, int] = (128, 128)     
    batch_size: int = 32
    seed: int = 42


    splits: Tuple[str, ...] = ("Train/Train", "Test/Test", "Validation/Validation")

    use_augmentation: bool = True


class PlantDiseaseDataModule:
   
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.base_dir = Path(cfg.data_root)

        self.splits = list(cfg.splits)
        self.split_paths: Dict[str, Path] = {s: self.base_dir / s for s in self.splits}

        self.df: Optional[pd.DataFrame] = None
        self.class_names: List[str] = []

        self.data_augmentation = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.RandomContrast(0.1),
            ],
            name="data_augmentation",
        )

        sns.set(style="whitegrid")

    # ---------------------------
    ## inspect structure
    # ---------------------------
    def print_base_dir_contents(self) -> None:
        print("Base directory:", self.base_dir)
        if not self.base_dir.exists():
            print("[ERROR] base_dir does not exist.")
            return
        print("Contents:")
        for item in self.base_dir.iterdir():
            print("  ", item.name, "| Dir:", item.is_dir())

    def inspect_splits_and_classes(self) -> Dict[str, List[str]]:
        info = {}
        for split in self.splits:
            split_path = self.split_paths[split]
            if not split_path.exists():
                print(f"[WARN] Split folder missing: {split_path}")
                info[split] = []
                continue

            classes = sorted([d.name for d in split_path.iterdir() if d.is_dir()])
            info[split] = classes
            print(f"\n=== {split} ===")
            print("Path:", split_path)
            print("Classes:", classes)
        return info

    # -----------------------------------------
    #  build dataframe with basic metadata
    # -----------------------------------------
    def build_dataframe(self) -> pd.DataFrame:
        records = []
        for split in self.splits:
            split_path = self.split_paths[split]
            if not split_path.exists():
                continue

            for cls_dir in split_path.iterdir():
                if not cls_dir.is_dir():
                    continue
                label = cls_dir.name

                for img_path in cls_dir.glob("*"):
                    if not img_path.is_file():
                        continue
                    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                        continue

                    records.append(
                        {
                            "split": split,
                            "label": label,
                            "filepath": str(img_path),
                            "filename": img_path.name,
                            "filesize_kb": img_path.stat().st_size / 1024.0,
                        }
                    )

        df = pd.DataFrame(records)
        print("Total images found:", len(df))

        train_subset = df[df["split"] == "Train/Train"]
        self.class_names = sorted(train_subset["label"].unique().tolist()) if not train_subset.empty else sorted(df["label"].unique().tolist())

        self.df = df
        return df

    # ---------------------------------------------------
    # class balance table
    # ---------------------------------------------------
    def class_balance_table(self) -> pd.DataFrame:
        assert self.df is not None, "Run build_dataframe() first."
        return self.df.groupby(["split", "label"]).size().unstack(fill_value=0)

    def plot_class_balance_per_split(self) -> None:
        assert self.df is not None, "Run build_dataframe() first."

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, split in zip(axes, self.splits):
            subset = self.df[self.df["split"] == split]
            if subset.empty:
                ax.set_title(f"{split} (no data)")
                ax.axis("off")
                continue

            counts = subset["label"].value_counts().sort_index()
            sns.barplot(x=counts.index, y=counts.values, ax=ax)
            ax.set_title(f"{split} class counts")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------
    #  image modes, width/height , corrupt detection
    # -----------------------------------------------------
    def add_image_shape_and_mode(self) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
        """
        Adds columns: width, height, mode
        Returns: (df_with_cols, corrupt_files[(filepath, error)])
        """
        assert self.df is not None, "Run build_dataframe() first."

        shapes = []
        corrupt_files: List[Tuple[str, str]] = []

        for _, row in self.df.iterrows():
            img_path = row["filepath"]
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    mode = img.mode
                shapes.append((width, height, mode))
            except Exception as e:
                corrupt_files.append((img_path, str(e)))
                shapes.append((np.nan, np.nan, None))

        self.df["width"], self.df["height"], self.df["mode"] = zip(*shapes)

        print("Unique image modes:")
        print(self.df["mode"].value_counts(dropna=False))
        print("\nNumber of corrupt/unreadable files:", len(corrupt_files))
        if corrupt_files:
            print("Example corrupt file:", corrupt_files[0])

        return self.df, corrupt_files

    # ---------------------------------------------------
    # shape stats & aspect ratio hist per split
    # ---------------------------------------------------
    def shape_stats(self) -> pd.DataFrame:
        assert self.df is not None, "Run build_dataframe() first."
        return self.df[["width", "height"]].describe().round(2)

    def top_shape_counts(self, top_n: int = 10) -> pd.Series:
        assert self.df is not None, "Run build_dataframe() first."
        shape_counts = self.df.groupby(["width", "height"]).size().sort_values(ascending=False)
        return shape_counts.head(top_n)

    def plot_aspect_ratio_hist_per_split(self, bins: int = 20) -> None:
        assert self.df is not None, "Run build_dataframe() first."
        self.df["aspect_ratio"] = self.df["width"] / self.df["height"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, split in zip(axes, self.splits):
            subset = self.df[(self.df["split"] == split) & self.df["aspect_ratio"].notna()]
            if subset.empty:
                ax.set_title(f"{split} (no data)")
                ax.axis("off")
                continue

            ax.hist(subset["aspect_ratio"], bins=bins)
            ax.set_title(f"{split} aspect ratio")
            ax.set_xlabel("width / height")
        axes[0].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------
    #  file size dist with readable ticks
    # ---------------------------------------------------
    def plot_filesize_distribution(self, bins: int = 30, tick_step_kb: int = 200) -> None:
        assert self.df is not None, "Run build_dataframe() first."

        fig, ax = plt.subplots()
        sns.histplot(self.df["filesize_kb"], bins=bins, ax=ax)
        ax.set_title("File size distribution (KB) - all splits")
        ax.set_xlabel("File size (KB)")

        max_size = float(self.df["filesize_kb"].max())
        ticks = np.arange(0, max_size + tick_step_kb, tick_step_kb)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(t)}" for t in ticks], rotation=45)

        plt.tight_layout()
        plt.show()

        print("File size stats (KB):")
        display(self.df["filesize_kb"].describe().round(2))

    # ---------------------------------------------------
    #  color stats on a sample
    # ---------------------------------------------------
    @staticmethod
    def _compute_image_stats(img_path: str):
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        mean_rgb = flat.mean(axis=0)
        std_rgb = flat.std(axis=0)
        brightness = flat.mean() 
        return mean_rgb, std_rgb, float(brightness)

    def compute_color_brightness_df(self, sample_n: int = 300) -> pd.DataFrame:
        assert self.df is not None, "Run build_dataframe() first."

        sample_df = self.df.sample(n=min(sample_n, len(self.df)), random_state=self.cfg.seed)

        stats_records = []
        for _, row in sample_df.iterrows():
            try:
                mean_rgb, std_rgb, brightness = self._compute_image_stats(row["filepath"])
                stats_records.append(
                    {
                        "split": row["split"],
                        "label": row["label"],
                        "mean_r": float(mean_rgb[0]),
                        "mean_g": float(mean_rgb[1]),
                        "mean_b": float(mean_rgb[2]),
                        "std_r": float(std_rgb[0]),
                        "std_g": float(std_rgb[1]),
                        "std_b": float(std_rgb[2]),
                        "brightness": float(brightness),
                    }
                )
            except Exception:
                
                continue

        color_df = pd.DataFrame(stats_records)
        return color_df

    # ---------------------------------------------------
    #  boxplots brightness + mean RGB by label
    # ---------------------------------------------------
    def plot_brightness_and_rgb_boxplots(self, color_df: pd.DataFrame) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.boxplot(data=color_df, x="label", y="brightness", ax=axes[0])
        axes[0].set_title("Brightness distribution by class")
        axes[0].set_xlabel("Class")
        axes[0].set_ylabel("Brightness (0–1)")

        melted = color_df.melt(
            id_vars=["label"],
            value_vars=["mean_r", "mean_g", "mean_b"],
            var_name="channel",
            value_name="mean_value",
        )
        sns.boxplot(data=melted, x="label", y="mean_value", hue="channel", ax=axes[1])
        axes[1].set_title("Mean RGB channel values by class")
        axes[1].set_xlabel("Class")
        axes[1].set_ylabel("Mean value (0–1)")
        axes[1].legend(title="Channel")

        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------
    #  duplicates via MD5
    # ---------------------------------------------------
    def find_exact_duplicates_md5(self) -> List[Tuple[str, str]]:
        assert self.df is not None, "Run build_dataframe() first."

        hash_to_paths = {}
        duplicates = []

        for fp in self.df["filepath"]:
            with open(fp, "rb") as f:
                filehash = hashlib.md5(f.read()).hexdigest()
            if filehash in hash_to_paths:
                duplicates.append((fp, hash_to_paths[filehash]))
            else:
                hash_to_paths[filehash] = fp

        return duplicates

    # ---------------------------------------------------
    #  sample raw images BEFORE preprocessing
    # ---------------------------------------------------
    def show_samples_before_preprocessing(self, split: str, samples_per_class: int = 3) -> None:
        assert self.df is not None, "Run build_dataframe() first."

        subset = self.df[self.df["split"] == split]
        if subset.empty:
            print(f"No data for split: {split}")
            return

        classes = sorted(subset["label"].unique())
        n_rows = len(classes)
        n_cols = samples_per_class

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

      
        if n_rows == 1:
            axes = np.expand_dims(axes, 0)
        if n_cols == 1:
            axes = np.expand_dims(axes, 1)

        for row_idx, cls in enumerate(classes):
            cls_subset = subset[subset["label"] == cls]
            sample_paths = cls_subset.sample(
                n=min(samples_per_class, len(cls_subset)),
                random_state=self.cfg.seed
            )["filepath"].tolist()

            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                ax.axis("off")
                if col_idx < len(sample_paths):
                    img_path = sample_paths[col_idx]
                    img = Image.open(img_path)
                    ax.imshow(img)
                    if col_idx == 0:
                        ax.set_ylabel(cls, fontsize=12)
                else:
                    ax.set_visible(False)

        plt.suptitle(f"{split} samples BEFORE preprocessing", fontsize=14)
        plt.tight_layout()
        plt.show()

    # ==========================================================
    #  tf.data pipeline for experiments 
    # ==========================================================
    def _label_to_index_map(self) -> Dict[str, int]:
        
        if self.class_names:
            return {c: i for i, c in enumerate(self.class_names)}
      
        assert self.df is not None
        classes = sorted(self.df["label"].unique())
        return {c: i for i, c in enumerate(classes)}

    def _load_and_resize(self, path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32) 
        img = tf.image.resize(img, self.cfg.img_size, method="bicubic")
        return img, label

    def build_tf_dataset(self, split: str, augment: bool = False, shuffle: bool = False) -> tf.data.Dataset:
        """
        Build tf.data for a given split. Use inside each model cell.
        """
        assert self.df is not None, "Run build_dataframe() first."
        subset = self.df[self.df["split"] == split].copy()
        if subset.empty:
            raise ValueError(f"No rows for split={split}")

        label_map = self._label_to_index_map()
        subset["label_idx"] = subset["label"].map(label_map)

        paths = subset["filepath"].values
        labels = subset["label_idx"].values

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(subset), seed=self.cfg.seed, reshuffle_each_iteration=True)

        ds = ds.map(self._load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)

        if augment and self.cfg.use_augmentation:
            def _aug(x, y):
                x = self.data_augmentation(x, training=True)
                return x, y
            ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(self.cfg.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
