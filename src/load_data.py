# load_data.py

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "languages.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "languages_clean.csv"

LABEL_COL = "Degree of endangerment"   # The target we're trying to predict
SPEAKER_COL = "Number of speakers"     # Most important numeric feature
NAME_COL = "Name in English"           # Language name (not used as feature)
COUNTRY_COL = "Countries"              # Country/region (used as feature)
# FAMILY_COL = "Language family"    not included in dataset         # Linguistic family (used as feature)
LAT_COL = "Latitude"                   # Geographic coordinates
LON_COL = "Longitude"


def load_raw() -> pd.DataFrame:
    
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"\n\nCould not find the dataset at:\n  {RAW_DATA_PATH}\n\n"
            "Please download it from:\n"
            "  https://www.kaggle.com/datasets/the-guardian/extinct-languages\n\n"
            "Then save it as:  data/raw/languages.csv"
        )

    
    df = pd.read_csv(RAW_DATA_PATH, encoding="latin-1")


    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Expected column '{LABEL_COL}' not found in CSV.\n"
            f"Columns found: {list(df.columns)}\n"
        )

    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns from {RAW_DATA_PATH.name}")
    return df


def load_processed() -> pd.DataFrame:
   
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DATA_PATH}.\n"
            "Run notebook 01_explore.ipynb first (it calls preprocess.save_processed)."
        )

    df = pd.read_csv(PROCESSED_DATA_PATH, encoding="latin-1")
    print(f"Loaded processed data: {len(df):,} rows, {len(df.columns)} columns")
    return df

if __name__ == "__main__":
    df = load_raw()
    print(df.shape)
    print(df.head())