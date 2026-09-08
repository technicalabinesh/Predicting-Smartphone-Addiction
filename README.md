# Predicting Smartphone Addiction

This repository contains a machine learning project focused on predicting smartphone addiction using an ensemble of gradient boosting models and a PyTorch tabular neural network.

## Project Overview

The main notebook performs:
- Data loading from Kaggle Playground Series S6E8
- Feature engineering (statistical, transformed, interaction, and group-based features)
- 10-fold stratified cross-validation
- Training of multiple models:
  - LightGBM
  - CatBoost
  - XGBoost
  - PyTorch Tabular ResNet
- Rank-based ensemble weight optimization using SciPy
- Submission file generation

## Repository Structure

- `Predicting Smartphone Addiction` — main Jupyter notebook/script containing the full pipeline
- `README.md` — project documentation

## Requirements

Install the core dependencies used in the notebook:

- Python 3.10+
- numpy
- pandas
- scipy
- scikit-learn
- lightgbm
- catboost
- xgboost
- torch

You can install them with:

```bash
pip install numpy pandas scipy scikit-learn lightgbm catboost xgboost torch
```

## How to Run

1. Download the competition data from Kaggle:
   - Playground Series S6E8
2. Update input file paths in the notebook if needed.
3. Run the notebook end-to-end.
4. The output submission file is generated as:
   - `submission_0.99_ensemble.csv`

## Notes

- The workflow is optimized for GPU-backed training where available.
- Random seeds are set for reproducibility.
- Model performance is evaluated with ROC-AUC.

## License

This project is shared for educational and experimentation purposes.
