"""Activity 4 - PCA with three scaling/transformation approaches.

1. log10 transformation + mean centering
2. autoscaling (mean centering, divide by standard deviation)
3. Pareto scaling (mean centering, divide by sqrt of standard deviation)

For each: scree plot, score plot (PC1 vs PC2) and loading plot, plus a
summary table comparing the models.
Run from the repository root:  python src/pca_analysis.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

CLEAN_PATH = Path("data/wine_clean.csv")
RESULTS = Path("results")
CLASS_COL = "class"
N_COMPONENTS = 10
CLASS_COLORS = {1: "#1f6f8b", 2: "#d1802f", 3: "#5b8c3a"}

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})


def log_mean_center(X):
    logged = np.log10(X)  # all values are > 0 after cleaning
    return logged - logged.mean()


def autoscale(X):
    return (X - X.mean()) / X.std(ddof=1)


def pareto(X):
    return (X - X.mean()) / np.sqrt(X.std(ddof=1))


SCALINGS = {
    "log_meancenter": ("Log10 + mean centering", log_mean_center),
    "autoscaling": ("Autoscaling", autoscale),
    "pareto": ("Pareto scaling", pareto),
}


def plot_model(key, title, pca, scores, loadings, y):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # Scree plot
    ev = pca.explained_variance_ratio_ * 100
    pcs = np.arange(1, len(ev) + 1)
    axes[0].bar(pcs, ev, color="#1f6f8b")
    axes[0].plot(pcs, np.cumsum(ev), "o-", color="#333333", ms=3,
                 label="Cumulative")
    axes[0].set_xticks(pcs)
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("Scree plot")
    axes[0].legend(frameon=False)

    # Score plot
    for c in sorted(np.unique(y)):
        m = y == c
        axes[1].scatter(scores[m, 0], scores[m, 1], s=18, alpha=0.8,
                        color=CLASS_COLORS[c], label=f"Class {c}")
    axes[1].axhline(0, color="#bbbbbb", lw=0.6)
    axes[1].axvline(0, color="#bbbbbb", lw=0.6)
    axes[1].set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    axes[1].set_title("Score plot")
    axes[1].legend(frameon=False)

    # Loading plot
    axes[2].scatter(loadings["PC1"], loadings["PC2"], s=14, color="#c0392b")
    for var, row in loadings.iterrows():
        axes[2].annotate(var, (row["PC1"], row["PC2"]), fontsize=7,
                         xytext=(3, 3), textcoords="offset points")
    axes[2].axhline(0, color="#bbbbbb", lw=0.6)
    axes[2].axvline(0, color="#bbbbbb", lw=0.6)
    axes[2].set_xlabel("PC1 loading")
    axes[2].set_ylabel("PC2 loading")
    axes[2].set_title("Loading plot")

    fig.suptitle(f"PCA - {title}", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / f"pca_{key}.png", dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(CLEAN_PATH)
    y = df[CLASS_COL].to_numpy()
    X = df.drop(columns=CLASS_COL)

    summary = []
    for key, (title, scale) in SCALINGS.items():
        Xs = scale(X)
        pca = PCA(n_components=N_COMPONENTS).fit(Xs)
        scores = pca.transform(Xs)
        loadings = pd.DataFrame(pca.components_[:2].T, index=X.columns,
                                columns=["PC1", "PC2"])
        loadings.round(3).to_csv(RESULTS / f"loadings_{key}.csv")
        plot_model(key, title, pca, scores, loadings, y)

        ev = pca.explained_variance_ratio_ * 100
        top = lambda pc: ", ".join(loadings[pc].abs().nlargest(3).index)
        summary.append({
            "scaling": title,
            "PC1_%": round(ev[0], 1),
            "PC2_%": round(ev[1], 1),
            "PC1+PC2_%": round(ev[0] + ev[1], 1),
            # class separation in the PC1-PC2 plane (-1 worst, 1 best)
            "silhouette_PC1_PC2": round(silhouette_score(scores[:, :2], y), 3),
            "top_PC1": top("PC1"),
            "top_PC2": top("PC2"),
        })

    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS / "pca_comparison.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
