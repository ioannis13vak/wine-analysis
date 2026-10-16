"""Activity 2 - Data loading and cleaning.

Reads data/wine_data.csv, checks structure, handles missing and erroneous
values and writes data/wine_clean.csv.
Run from the repository root:  python src/data_cleaning.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/wine_data.csv")
CLEAN_PATH = Path("data/wine_clean.csv")
CLASS_COL = "class"
# Plausible physical ranges used to flag erroneous values
VALID_RANGES = {
    "alcohol": (8, 16),        # % vol
    "ash": (1, 4),
    "magnesium": (50, 200),    # mg/L
}


def main():
    # 1. Load
    df = pd.read_csv(RAW_PATH)

    # 2. Dimensions and structure
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(df.dtypes.to_string())
    print("Class counts:\n", df[CLASS_COL].value_counts().sort_index().to_string())

    # Duplicate rows are an inconsistency in their own right
    n_dup = int(df.duplicated().sum())
    print(f"Duplicate rows: {n_dup}")
    df = df.drop_duplicates().reset_index(drop=True)

    # 5-6. Erroneous values -> set to NaN so they are imputed like missing values
    features = [c for c in df.columns if c != CLASS_COL]
    neg = (df[features] < 0).sum()
    df[features] = df[features].mask(df[features] < 0)
    for col, (lo, hi) in VALID_RANGES.items():
        bad = ~df[col].between(lo, hi) & df[col].notna()
        if bad.any():
            print(f"Out-of-range {col}: {df.loc[bad, col].tolist()} -> set to NaN")
        df.loc[bad, col] = np.nan
    print("Negative values found:", int(neg.sum()))

    # 3. Missing values (original + values flagged above)
    missing = df[features].isna().sum()
    print("Missing values per variable:\n", missing[missing > 0].to_string())
    n_imputed = int(missing.sum())
    print(f"Values to impute: {n_imputed} "
          f"({100 * n_imputed / df[features].size:.2f}% of all cells)")

    # 4. Impute with the class-wise median: robust to outliers and keeps
    #    differences between wine classes intact
    df[features] = df.groupby(CLASS_COL)[features].transform(
        lambda s: s.fillna(s.median())
    )
    assert df.isna().sum().sum() == 0

    # 7. Save
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved {CLEAN_PATH} with shape {df.shape}")


if __name__ == "__main__":
    main()
