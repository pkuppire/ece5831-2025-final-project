import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageChops

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

MODELS = Path("models")


def autocrop(im: Image.Image, bg=None, pad=6):
   
    if im.mode != "RGB":
        im = im.convert("RGB")

    if bg is None:
        bg = im.getpixel((0, 0))

    bg_img = Image.new("RGB", im.size, bg)
    diff = ImageChops.difference(im, bg_img)
    bbox = diff.getbbox()
    if bbox is None:
        return im  

    left, upper, right, lower = bbox
    left = max(0, left - pad)
    upper = max(0, upper - pad)
    right = min(im.size[0], right + pad)
    lower = min(im.size[1], lower + pad)
    return im.crop((left, upper, right, lower))

def load_curve(exp, crop=True):
    im = Image.open(MODELS / exp / "training_curves.png")
    return autocrop(im, pad=8) if crop else im

def plot_cm(ax, cm, title):
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["Healthy", "Powdery", "Rust"], fontsize=9)
    ax.set_yticklabels(["Healthy", "Powdery", "Rust"], fontsize=9)

    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center", fontsize=9)

    ax.set_xlabel("Predicted", fontsize=9, labelpad=4)
    ax.set_ylabel("True", fontsize=9, labelpad=4)

def save_fig(path):
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()

fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)

for ax, exp, title in zip(
    axs,
    ["baseline_cnn_128", "baseline_cnn_224", "improved_cnn_128"],
    ["Baseline CNN @128", "Baseline CNN @224", "Improved CNN @128"]
):
    ax.imshow(load_curve(exp, crop=True))
    ax.set_title(title, fontsize=11, pad=6)
    ax.axis("off")

save_fig(FIG_DIR / "cnn_curves.png")

fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)

for ax, exp, title in zip(
    axs,
    ["vgg_pretrained_224", "resnet_pretrained_224", "efficientnetb0_tl_224"],
    ["VGG (Frozen)", "ResNet50 (Frozen)", "EfficientNetB0 (Frozen)"]
):
    ax.imshow(load_curve(exp, crop=True))
    ax.set_title(title, fontsize=11, pad=6)
    ax.axis("off")

save_fig(FIG_DIR / "tl_frozen_curves.png")

fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)

for ax, exp, title in zip(
    axs,
    ["vgg_finetuned_224", "resnet_finetuned_224", "efficientnetb0_ft_224_last20_lr1e5"],
    ["VGG (Fine-tuned)", "ResNet50 (Fine-tuned)", "EfficientNetB0 (Fine-tuned)"]
):
    ax.imshow(load_curve(exp, crop=True))
    ax.set_title(title, fontsize=11, pad=6)
    ax.axis("off")

save_fig(FIG_DIR / "tl_finetuned_curves.png")

cm_baseline = np.array([[49,1,0],[5,45,0],[2,0,48]])
cm_best = np.array([[50,0,0],[2,48,0],[0,1,49]])

fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.02)

plot_cm(axs[0], cm_baseline, "Baseline CNN @128")
plot_cm(axs[1], cm_best, "Best Fine-tuned Model")

save_fig(FIG_DIR / "cm_compare.png")
