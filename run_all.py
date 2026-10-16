"""Run the full pipeline: raw data -> cleaning -> EDA -> scaling + PCA -> results."""
import runpy

for step in ["src/data_cleaning.py", "src/descriptive_stats.py",
             "src/visualization.py", "src/pca_analysis.py"]:
    print(f"\n=== {step} ===")
    runpy.run_path(step, run_name="__main__")
