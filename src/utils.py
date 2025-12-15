from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_history(history_dict: Dict[str, List[float]], path: Path) -> None:
    save_json(history_dict, path)

def load_history(path: Path) -> Dict[str, List[float]]:
    return load_json(path)


# --------------------------
# Plot history
# --------------------------
def plot_history(history_dict: Dict[str, List[float]], title: str = "", save_path: Optional[Path] = None) -> None:
    plt.figure(figsize=(12, 4))
    plt.suptitle(title)

    # Accuracy
    plt.subplot(1, 2, 1)
    if "accuracy" in history_dict:
        plt.plot(history_dict["accuracy"], marker="o", label="Train acc")
    if "val_accuracy" in history_dict:
        plt.plot(history_dict["val_accuracy"], marker="o", label="Val acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    if "loss" in history_dict:
        plt.plot(history_dict["loss"], marker="o", label="Train loss")
    if "val_loss" in history_dict:
        plt.plot(history_dict["val_loss"], marker="o", label="Val loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    if save_path:
        ensure_dir(save_path.parent)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# --------------------------
# Confusion matrix + report
# --------------------------
def eval_predictions(model: tf.keras.Model, ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    for x, y in ds:
        probs = model.predict(x, verbose=0)
        y_true.append(y.numpy())
        y_pred.append(np.argmax(probs, axis=1))
    return np.concatenate(y_true), np.concatenate(y_pred)

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.grid(False)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12
            )

    plt.tight_layout()
    plt.show()

def evaluate_and_report(model: tf.keras.Model, ds_test: tf.data.Dataset, class_names: List[str], model_name: str = ""):
    loss, acc = model.evaluate(ds_test, verbose=0)
    y_true, y_pred = eval_predictions(model, ds_test)

    print(f"\n=== {model_name} ===")
    print(f"Test loss: {loss:.4f} | Test acc: {acc:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    plot_confusion_matrix(y_true, y_pred, class_names, title=f"{model_name} - Confusion Matrix")

    return loss, acc, y_true, y_pred


# --------------------------
# Wrong predictions grid
# --------------------------
def show_wrong_predictions(
    model: tf.keras.Model,
    ds_test: tf.data.Dataset,
    class_names: List[str],
    max_images: int = 12
) -> None:
    wrong_imgs, wrong_titles = [], []

    for x_batch, y_batch in ds_test:
        probs = model.predict(x_batch, verbose=0)
        preds = np.argmax(probs, axis=1)
        yb = y_batch.numpy()

        for i in range(len(yb)):
            if preds[i] != yb[i]:
                wrong_imgs.append(x_batch[i].numpy())
                wrong_titles.append(f"T:{class_names[int(yb[i])]}\nP:{class_names[int(preds[i])]}")
                if len(wrong_imgs) >= max_images:
                    break
        if len(wrong_imgs) >= max_images:
            break

    if not wrong_imgs:
        print("No wrong predictions found.")
        return

    cols = 4
    rows = int(np.ceil(len(wrong_imgs) / cols))
    plt.figure(figsize=(cols * 3.2, rows * 3.2))

    for idx, (img, title) in enumerate(zip(wrong_imgs, wrong_titles), start=1):
        plt.subplot(rows, cols, idx)
        plt.imshow(np.clip(img, 0, 1))
        plt.title(title, fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
