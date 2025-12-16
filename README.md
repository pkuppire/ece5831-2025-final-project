# ece5831-2025-final-project
# Plant Disease Classification Project

This project implements CNN-based models to classify plant leaf images into three categories: **Healthy**, **Powdery**, and **Rust**.  
The project is designed with a **clear, modular structure**, and all experiments can be reproduced from a **final notebook**.

---
## Project Structure

```text
ECE5831-2025-FINAL-PROJECT/
├── data/
│   └── plant-disease-recognition-dataset/
│       ├── Train/
│       │   ├── Healthy/
│       │   ├── Powdery/
│       │   └── Rust/
│       ├── Validation/
│       │   ├── Healthy/
│       │   ├── Powdery/
│       │   └── Rust/
│       └── Test/
│           ├── Healthy/
│           ├── Powdery/
│           └── Rust/
│
├── models/
│   ├── baseline_cnn_128/
│   ├── baseline_cnn_224/
│   ├── improved_cnn_128/
│   ├── improved_cnn_224/
│   ├── efficientnetb0_tl_224/
│   ├── efficientnetb0_ft_light_224/
│   ├── resnet_pretrained_224/
│   ├── vgg_pretrained_224/
│   ├── vgg_finetuned_224/
│   ├── *.keras              # Saved trained models (ignored in git)
│   ├── history.json         # Training history
│   ├── meta.json            # Experiment configuration
│   ├── test_metrics.json    # Final test results
│   └── training_curves.png  # Accuracy & loss plots
│
├── figures/
│   ├── cnn_curves.png
│   ├── tl_frozen_curves.png
│   ├── tl_finetuned_curves.png
│   └── cm_compare.png
│
├── src/
│   ├── data_eda.py
│   ├── utils.py
│   ├── baseline_cnn.py
│   ├── improved_cnn_128.py
│   ├── improved_cnn_224.py
│   ├── pretrained_vgg16.py
│   ├── pretrained_vgg16_finetune.py
│   ├── pretrained_resnet50.py
│   ├── pretrained_resnet50_finetune.py
│   ├── pretrained_efficientnet_b0.py
│   └── pretrained_efficientnet_b0_finetune.py
│
├── make_report_figures.py
├── final_project.ipynb
└── README.md
```
---


## Dataset Setup

The dataset is organized into **Train**, **Validation**, and **Test** folders.  
Each split contains three class subfolders:
Healthy/
Powdery/
Rust/

---

## How the Project Is Designed

- All model logic is implemented as **separate Python files** inside `src/`
- Each model saves its results automatically inside a corresponding folder in `models/`
- The **final_project.ipynb** notebook:
  - Loads data
  - Imports model classes
  - Trains or loads models
  - Plots training curves
  - Evaluates on the test set

This design keeps the notebook clean and readable!

---

## How to Run the Project

### 1. Select the Correct Python Environment

This project was developed and tested using the **same Conda environment used throughout the semester**: ece-5831-2025

Before running the notebook:
- Open `final_project.ipynb`
- Select the kernel **ece-5831-2025**
- Restart the kernel to ensure a clean state

No additional environment setup is required.

---

### 2. Run the Final Notebook
final_project.ipynb

Run the notebook **cell by cell, from top to bottom**.

The notebook will:
- Load the dataset
- Import model implementations from `src/`
- Load pretrained or previously trained models
- Evaluate models on the test set
- Display training curves, metrics, and confusion matrices

---

### 3. Avoid Re-training Models 

To ensure reproducible execution:

- For **all model experiments**, set: retrain = False

This will load existing trained models from the models/ directory instead of retraining them.



## Links

Link for Presentation video: https://youtu.be/detW35HMX7s?si=_GTa5HPkzxCJ6twL 

Link for demo video : https://youtu.be/7m3CUkdC2Ek?si=DqfEirCRlp4Pnzuj 

Link for presentation slides: https://docs.google.com/presentation/d/13eUgs199TmuqNG16QlGHWP5xdWCfiAi7/edit?usp=sharing&ouid=112501891501364681430&rtpof=true&sd=true

Link for Report: https://drive.google.com/file/d/1_KV-loHZH0JRQrcLFhqZDe54Ij9Kcu--/view?usp=sharing

Link fot Dataset:  https://drive.google.com/drive/folders/1wtTPqgwsdYKPq6yu4rqwR5OY0UdZym8X?usp=sharing

Link for Kaggle Dataset: https://www.kaggle.com/datasets/rashikrahmanpritom/plant-disease-recognition-dataset

