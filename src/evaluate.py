# evaluate.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from src.preprocess import ENDANGERMENT_ORDER


def evaluate_model(model, X_test, y_test, label_order=None) -> dict:
  
    if label_order is None:
        label_order = sorted(y_test.unique())

    
    y_pred = model.predict(X_test)

    
    acc = accuracy_score(y_test, y_pred)

    
    labels_present = [l for l in label_order if l in y_test.unique()]
    cm = confusion_matrix(y_test, y_pred, labels=labels_present)

   
    report_str = classification_report(
        y_test, y_pred,
        labels=labels_present,
        zero_division=0    
    )

   
    report_df = classification_report_to_df(y_test, y_pred, labels=labels_present)

    
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0, labels=labels_present)

    return {
        "y_pred": y_pred,
        "cm": cm,
        "cm_labels": labels_present,
        "report": report_str,
        "report_df": report_df,
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


def print_classification_report(results: dict) -> None:
  
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall accuracy: {results['accuracy']:.3f}")
    print("  ↑ This number is misleading for imbalanced data!")
    print("    A model that always predicts 'Safe' could get similar accuracy.")

    print(f"\nMacro F1:         {results['macro_f1']:.3f}")
    print("  ↑ This is the number that matters.")
    print("    It averages F1 across all classes equally — minority classes included.")

    print("\n--- Per-class classification report ---")
    print(results["report"])
    print("\nHow to read this:")
    print("  precision  = of all times we predicted class X, how often we were right")
    print("  recall     = of all actual class X, how many we found (sensitivity)")
    print("  f1-score   = harmonic mean of precision and recall (the balanced score)")
    print("  support    = number of actual samples in this class in the test set")
    print("\n  macro avg    = treats all classes equally (what you should focus on)")
    print("  weighted avg = weighted by sample count (inflates majority class scores)")


def classification_report_to_df(y_test, y_pred, labels) -> pd.DataFrame:
  
    report_dict = {}
    for label in labels:
        mask_true = y_test == label
        mask_pred = np.array(y_pred) == label

        tp = (mask_true & mask_pred).sum()
        fp = (~mask_true & mask_pred).sum()
        fn = (mask_true & ~mask_pred).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)

        report_dict[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": mask_true.sum(),
        }

    return pd.DataFrame(report_dict).T


def compare_models(results_dict: dict) -> pd.DataFrame:
   
    rows = []
    for model_name, results in results_dict.items():
        row = {"model": model_name, "macro_f1": round(results["macro_f1"], 3)}
        # Add per-class F1
        for cls, metrics in results["report_df"].iterrows():
            row[f"f1_{cls}"] = metrics["f1"]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    return df
