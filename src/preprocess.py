# preprocess.py

import pandas as pd
import numpy as np
from pathlib import Path
from src.load_data import (
    LABEL_COL, SPEAKER_COL, COUNTRY_COL, LAT_COL, LON_COL,
    PROCESSED_DATA_PATH
)


ENDANGERMENT_ORDER = [
    "Safe",
    "Vulnerable",
    "Definitely endangered",
    "Severely endangered",
    "Critically endangered",
    "Extinct",
]


NUMERIC_FEATURES = [
    SPEAKER_COL,   # Number of speakers — the most predictive feature
    LAT_COL,       # Latitude — geography correlates with endangerment patterns
    LON_COL,       # Longitude
]

CATEGORICAL_FEATURES = [
    COUNTRY_COL,   # Country or region — political context matters
]


def clean(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()  # Never modify the original — always work on a copy

    before = len(df)
    df = df.dropna(subset=[LABEL_COL])
    after = len(df)
    print(f"Dropped {before - after} rows with missing labels. {after} rows remain.")

    df[LABEL_COL] = df[LABEL_COL].str.strip()

    unknown_labels = set(df[LABEL_COL].unique()) - set(ENDANGERMENT_ORDER)
    if unknown_labels:
        print(f"Warning: unexpected label values found: {unknown_labels}")
        print("These rows will be dropped.")
        df = df[df[LABEL_COL].isin(ENDANGERMENT_ORDER)]

    if SPEAKER_COL in df.columns:
        missing_speakers = df[SPEAKER_COL].isna().sum()
        df[SPEAKER_COL] = pd.to_numeric(df[SPEAKER_COL], errors="coerce").fillna(0)
        print(f"Filled {missing_speakers} missing speaker counts with 0.")
    else:
        df[SPEAKER_COL] = 0

   
    for coord_col in [LAT_COL, LON_COL]:
        if coord_col in df.columns:
            df[coord_col] = pd.to_numeric(df[coord_col], errors="coerce").fillna(0)
        else:
            df[coord_col] = 0.0

   
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].fillna("Unknown").str.strip()
        else:
            df[cat_col] = "Unknown"

    return df


def get_X_y(df: pd.DataFrame):

    
    y = df[LABEL_COL]

    
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    X_numeric = df[numeric_cols].copy()

    
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    if cat_cols:
      
        X_cat = pd.get_dummies(df[cat_cols], drop_first=True)
    else:
        X_cat = pd.DataFrame(index=df.index)

    X = pd.concat([X_numeric, X_cat], axis=1)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution:\n{y.value_counts()}")

    return X, y


def save_processed(df: pd.DataFrame) -> None:
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False, encoding="latin-1")
    print(f"Saved processed data to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    from src.load_data import load_raw          # load the raw CSV first
    
    df_raw = load_raw()                         # get the DataFrame
    df_clean = clean(df_raw)                    # now pass it into clean()
    
    print("\nCleaned DataFrame shape:", df_clean.shape)
    print("\nFirst 3 rows:")
    print(df_clean.head(3))
    
    X, y = get_X_y(df_clean)                   # extract features and labels
    print("\nX shape:", X.shape)
    print("y value counts:\n", y.value_counts())
    save_processed(df_clean)