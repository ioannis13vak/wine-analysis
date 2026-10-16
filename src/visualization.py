"""Activity 3b - Exploratory figures, saved to results/.

Run from the repository root:  python src/visualization.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

CLEAN_PATH = Path("data/wine_clean.csv")
RESULTS = Path("results")
CLASS_COL = "class"
CLASS_COLORS = {1: "#1f6f8b", 2: "#d1802f", 3: "#5b8c3a"}
SELECTED = ["flavanoids", "proline", "color_intensity", "alcohol"]

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})


def histograms(df, features):
    fig, axes = plt.subplots(4, 4, figsize=(12, 10))
    for ax, col in zip(axes.flat, features):
        ax.hist(df[col], bins=20, color="#1f6f8b", edgecolor="white")
        ax.set_title(col)
    for ax in axes.flat[len(features):]:
        ax.axis("off")
    fig.suptitle("Distribution of each variable (cleaned data)")
    fig.tight_layout()
    fig.savefig(RESULTS / "histograms.png", dpi=150)
    plt.close(fig)


def boxplots_by_class(df):
    fig, axes = plt.subplots(1, len(SELECTED), figsize=(13, 3.6))
    classes = sorted(df[CLASS_COL].unique())
    for ax, col in zip(axes, SELECTED):
        data = [df.loc[df[CLASS_COL] == c, col] for c in classes]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], classes):
            patch.set_facecolor(CLASS_COLORS[c]); patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(classes) + 1), [f"Class {c}" for c in classes])
        ax.set_title(col)
    fig.suptitle("Selected variables compared between wine classes")
    fig.tight_layout()
    fig.savefig(RESULTS / "boxplots_by_class.png", dpi=150)
    plt.close(fig)


def scatter(df):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for c, g in df.groupby(CLASS_COL):
        ax.scatter(g["flavanoids"], g["proline"], s=22, alpha=0.8,
                   color=CLASS_COLORS[c], label=f"Class {c}")
    ax.set_xlabel("Flavanoids"); ax.set_ylabel("Proline")
    ax.legend(frameon=False)
    ax.set_title("Flavanoids vs proline")
    fig.tight_layout()
    fig.savefig(RESULTS / "scatter_flavanoids_proline.png", dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(CLEAN_PATH)
    features = [c for c in df.columns if c != CLASS_COL]
    histograms(df, features)
    boxplots_by_class(df)
    scatter(df)
    print("Figures saved to", RESULTS)


if __name__ == "__main__":
    main()
