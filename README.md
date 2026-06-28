# LangVault

> Classify endangered languages from sparse features.  
> A hands-on ML project for learning **confusion matrices** and **class imbalance**.

---

## What this project teaches you

| Concept | Where you'll see it |
|---|---|
| Multiclass classification | All notebooks |
| Class imbalance (the core problem) | `02_baseline.ipynb` |
| Confusion matrix | `04_confusion_matrix.ipynb` |
| Precision / Recall / F1 | `02_baseline.ipynb`, `03_imbalance_fix.ipynb` |
| SMOTE oversampling | `03_imbalance_fix.ipynb` |
| Class weights in sklearn | `03_imbalance_fix.ipynb` |
| Random Forest classifier | `02_baseline.ipynb` |
| Stratified cross-validation | `03_imbalance_fix.ipynb` |

---

## Quick start

```bash
# 1. Clone or unzip this repo
cd LangVault

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
#    Go to: https://www.kaggle.com/datasets/the-guardian/extinct-languages
#    Download and put the CSV in:  data/raw/languages.csv


```

---

## Dataset

**Source:** Google Endangered Languages Project (via Kaggle / The Guardian)  
**File to download:** Put it at `data/raw/languages.csv`  
**Key column (label):** `Degree of endangerment`  
**Classes (UNESCO 5-tier scale):**
- Safe
- Vulnerable  
- Definitely endangered
- Severely endangered
- Critically endangered
- Extinct

This label distribution is **heavily skewed** — that's the whole point.

---

## Folder map

```
LangVault/
├── README.md                  ← you are here
├── ABOUT_THE_PROJECT.md       ← deep concepts explanation
├── requirements.txt
├── data/
│   ├── raw/                   ← put languages.csv here
│   └── processed/             ← cleaned data lands here automatically
├── notebooks/
│   ├── 01_explore.ipynb       ← EDA: understand the data
│   ├── 02_baseline.ipynb      ← naive model, see the imbalance problem
│   ├── 03_imbalance_fix.ipynb ← fix it with SMOTE + class weights
│   └── 04_confusion_matrix.ipynb ← deep dive on confusion matrix
├── src/
│   ├── load_data.py           ← loading utilities
│   ├── preprocess.py          ← cleaning + feature engineering
│   ├── train.py               ← model training
│   ├── evaluate.py            ← metrics and evaluation
│   └── visualise.py           ← all plots
└── outputs/
    ├── models/                ← saved models (.pkl)
    ├── figures/               ← saved plots
    └── reports/               ← saved classification reports
```

---


