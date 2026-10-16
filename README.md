# wine-analysis

Collaborative data-analysis practical using the wine dataset.

## Project description
This project builds a reproducible analysis pipeline for the wine dataset:
chemical analysis of 178 Italian wines from three cultivars using PCA.

## Repository structure
```
wine-analysis/
├── data/
│   ├── wine_data.csv        raw data (contains missing and erroneous values)
│   └── wine_clean.csv       output of data cleaning
├── src/
│   ├── data_cleaning.py     Activity 2
│   ├── descriptive_stats.py Activity 3 (statistics, class comparison)
│   ├── visualization.py     Activity 3 (figures)
│   └── pca_analysis.py      Activity 4 (3 scalings, scree/score/loading plots)
├── results/                 figures (.png) and tables (.csv)
├── run_all.py               runs the whole pipeline in order
├── requirements.txt
└── README.md
```

## Pipeline
Raw data -> data cleaning -> clean dataset -> exploratory analysis -> scaling -> PCA -> figures/results

## How to run
```bash
pip install -r requirements.txt
python run_all.py          # from the repository root
```
Each script can also be run on its own, e.g. `python src/pca_analysis.py`.

## Requirements
Python 3.10+ with numpy, pandas, matplotlib, scipy and scikit-learn (see requirements.txt).

## Contributors
| Component | Branch | Contributor |
|---|---|---|
| Data loading & cleaning | data-cleaning | G. Vakkas |
| Descriptive statistics | descriptive-stats | G. Vakkas |
| Data visualization | visualization | G. Vakkas |
| Scaling & PCA | pca-analysis | G. Vakkas |
