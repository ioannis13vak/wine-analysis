"""Activity 3a - Descriptive statistics of the cleaned wine data.

Run from the repository root:  python src/descriptive_stats.py
"""
from pathlib import Path

import pandas as pd

CLEAN_PATH = Path("data/wine_clean.csv")
RESULTS = Path("results")
CLASS_COL = "class"


def main():
    df = pd.read_csv(CLEAN_PATH)
    features = df.drop(columns=CLASS_COL)

    overall = features.agg(["mean", "median", "std", "min", "max"]).T.round(3)
    overall.to_csv(RESULTS / "descriptive_statistics.csv")
    print(overall.to_string())

    by_class = df.groupby(CLASS_COL).agg(["mean", "std"]).round(3)
    by_class.to_csv(RESULTS / "statistics_by_class.csv")
    print("\nClass means:\n", df.groupby(CLASS_COL).mean().round(2).T.to_string())


if __name__ == "__main__":
    main()
