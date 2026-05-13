## Overview
This repository contains the `further_models.py` pipeline in the models folder, which predicts chronological age from single-cell RNA-sequencing (scRNA-seq) and genotype data. The script trains two-stage machine learning ensembles across multiple specific immune cell populations, as well as an "overall" pseudobulk profile. 

By leveraging diverse data modalities—raw pseudobulk gene expression, Geneformer foundational model embeddings, and Genotype Principal Components (PCs)—the pipeline ensures that morphological, genetic, and transcriptomic signals are all captured to accurately estimate donor age.

The Jupyter notebook `models_stats_final.ipynb` runs the calculations for how well the models perform vs the true age in the test dataset.

## Model Architecture

![Model Architecture](model_workflow.png)

The pipeline employs a **Stacked Generalization (Stacking)** ensemble framework.

1. **Modality-Specific Base Models (Level 0):**
   * **Pseudobulk Expression + Sex:** XGBoost, LightGBM (HistGradientBoosting), Random Forest, and ElasticNet.
   * **Geneformer Embeddings + Sex:** Multi-Layer Perceptron (MLP) and K-Nearest Neighbors (KNN).
   * **Genotype PCs + Sex:** Ridge Regression and Support Vector Regression (SVR).
2. **Meta-Model (Level 1):**
   * The Out-of-Fold (OOF) predictions from all 8 base models are concatenated into a new feature matrix.
   * A **Ridge Regressor** (alpha=10.0) acts as the meta-model, learning the optimal weighting of the base models' predictions to output the final age.

This complete architecture is trained **independently** for 6 different cell-type configurations: `overall`, `CD4 T`, `CD8 T`, `NK`, `B cells`, and `monocytes`.

The results are found in the csv file `multi_celltype_ensemble_submission.csv`.

The statistics for the models are as follows:

| Model | MAE | RMSE | Pearson r | Spearman ρ | R² |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **age_overall** | 5.7508 | 6.9216 | 0.9163 | 0.8805 | 0.8369 |
| **age_CD4_T** | 6.5269 | 7.9591 | 0.8883 | 0.8503 | 0.7844 |
| **age_CD8_T** | 6.4520 | 7.6778 | 0.8967 | 0.8753 | 0.7993 |
| **age_NK** | 6.3051 | 7.8102 | 0.8918 | 0.8548 | 0.7924 |
| **age_B_cells** | 6.7603 | 8.3769 | 0.8766 | 0.8343 | 0.7611 |
| **age_monocytes** | 7.2100 | 8.7411 | 0.8623 | 0.8227 | 0.7399 |
