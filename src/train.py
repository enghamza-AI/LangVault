# train.py

import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score


from imblearn.over_sampling import SMOTE

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


RF_PARAMS = {
    "n_estimators": 200,       
                               
    "max_depth": None,          
    "min_samples_leaf": 2,      
    "random_state": 42,        
    "n_jobs": -1,               
}


def split_data(X, y, test_size: float = 0.2):
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,       
        random_state=42
    )
    print(f"Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


def train_baseline(X_train, y_train) -> RandomForestClassifier:
  
    print("Training baseline Random Forest (no imbalance handling)...")

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    print("Done. Note: this model will likely ignore minority classes.")
    return model


def train_class_weights(X_train, y_train) -> RandomForestClassifier:
   
    print("Training Random Forest with class_weight='balanced'...")

   
    params = {**RF_PARAMS, "class_weight": "balanced"}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    print("Done. Minority classes now carry higher weight in the loss.")
    return model


def train_smote(X_train, y_train):
   
    print("Applying SMOTE to training data...")

    
    smote = SMOTE(k_neighbors=3, random_state=42)

    # SMOTE needs numeric arrays, not DataFrames
    X_arr = X_train.values if hasattr(X_train, "values") else X_train
    y_arr = y_train.values if hasattr(y_train, "values") else y_train

    X_resampled, y_resampled = smote.fit_resample(X_arr, y_arr)

    # Show how the class distribution changed
    import pandas as pd
    before = pd.Series(y_arr).value_counts()
    after = pd.Series(y_resampled).value_counts()
    print("\nClass counts before SMOTE:")
    print(before.to_string())
    print("\nClass counts after SMOTE:")
    print(after.to_string())

    print("\nTraining Random Forest on SMOTE-balanced data...")
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_resampled, y_resampled)

    print("Done.")
    return model, X_resampled, y_resampled


def cross_validate_model(model, X, y, n_splits: int = 5) -> dict:
  
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

   
    scores = cross_val_score(model, X, y, cv=skf, scoring="f1_macro")

    print(f"Macro F1 across {n_splits} folds: {scores}")
    print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")

    return {"macro_f1_scores": scores, "mean_macro_f1": scores.mean()}


def save_model(model, name: str) -> Path:
   
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"Model saved to {path}")
    return path


def load_model(name: str):
    """Load a model previously saved with save_model()."""
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No model found at {path}. Train and save it first.")
    return joblib.load(path)
