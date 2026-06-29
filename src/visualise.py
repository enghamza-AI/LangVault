# visualise.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def _save(fig, name: str) -> None:
 
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {path}")


def plot_class_distribution(y, title: str = "Class distribution", save_name: str = None):
   
    counts = y.value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(counts.index, counts.values, color="#5B8DB8")

    # Add count labels on each bar so you can read exact numbers
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
            f"{count:,}", va="center", ha="left", fontsize=11
        )

    ax.set_xlabel("Number of languages")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)  # cleaner look
    plt.tight_layout()

    if save_name:
        _save(fig, save_name)

    plt.show()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    title: str = "Confusion matrix",
    normalise: bool = False,
    save_name: str = None
):
    
    if normalise:
        # Divide each row by the row sum so each row sums to 1 (= 100%)
        # This shows recall per class, accounting for class size
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.where(row_sums > 0, cm / row_sums, 0)
        fmt = ".0%"
        cbar_label = "Recall (fraction of true class correctly predicted)"
    else:
        cm_display = cm
        fmt = "d"
        cbar_label = "Number of samples"

    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        cm_display,
        annot=True,        # Show numbers inside each cell
        fmt=fmt,           # Format: "d" for integers, ".0%" for percentages
        cmap="Blues",      # Colour scale: light = low, dark = high
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,    # Grid lines between cells for readability
        cbar_kws={"label": cbar_label},
    )

    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)

    # Rotate x-axis labels so they don't overlap
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_name:
        _save(fig, save_name)

    plt.show()
    return fig


def plot_f1_comparison(comparison_df: pd.DataFrame, save_name: str = None):
   
    # Filter to just the per-class F1 columns (not overall macro_f1)
    f1_cols = [c for c in comparison_df.columns if c.startswith("f1_")]
    plot_labels = [c.replace("f1_", "") for c in f1_cols]

    data = comparison_df[f1_cols].copy()
    data.columns = plot_labels

    # Transpose: rows = classes, columns = models
    data_T = data.T

    fig, ax = plt.subplots(figsize=(12, 6))

    n_models = len(data_T.columns)
    n_classes = len(data_T)
    bar_width = 0.8 / n_models
    x = np.arange(n_classes)

    colours = ["#5B8DB8", "#E07B39", "#4BAF82"]  # one colour per model

    for i, model_name in enumerate(data_T.columns):
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            data_T[model_name],
            width=bar_width,
            label=model_name,
            color=colours[i % len(colours)],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(data_T.index, rotation=25, ha="right")
    ax.set_ylabel("F1 score (0 = worst, 1 = best)")
    ax.set_title("Per-class F1 comparison across models")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="0.5 reference")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_name:
        _save(fig, save_name)

    plt.show()
    return fig


def plot_feature_importance(model, feature_names: list, top_n: int = 20, save_name: str = None):
    
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    ax.barh(top_features[::-1], top_importances[::-1], color="#5B8DB8")
    ax.set_xlabel("Feature importance (mean decrease in Gini impurity)")
    ax.set_title(f"Top {top_n} features by importance")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_name:
        _save(fig, save_name)

    plt.show()
    return fig
