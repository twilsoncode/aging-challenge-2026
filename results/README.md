## Overview
This repository contains the `further_models.py` pipeline in the models folder, which predicts chronological age from single-cell RNA-sequencing (scRNA-seq) and genotype data. The script trains robust, two-stage machine learning ensembles across multiple specific immune cell populations, as well as an "overall" pseudobulk profile. 

By leveraging diverse data modalities—raw pseudobulk gene expression, Geneformer foundational model embeddings, and Genotype Principal Components (PCs)—the pipeline ensures that morphological, genetic, and transcriptomic signals are all captured to accurately estimate donor age.

## Model Architecture

![Model Architecture](model_worfklow.png)

The pipeline employs a **Stacked Generalization (Stacking)** ensemble framework.

1. **Modality-Specific Base Models (Level 0):**
   * **Pseudobulk Expression + Sex:** XGBoost, LightGBM (HistGradientBoosting), Random Forest, and ElasticNet. *(Includes variance thresholding and K-Best feature selection)*.
   * **Geneformer Embeddings + Sex:** Multi-Layer Perceptron (MLP) and K-Nearest Neighbors (KNN).
   * **Genotype PCs + Sex:** Ridge Regression and Support Vector Regression (SVR).
2. **Meta-Model (Level 1):**
   * The Out-of-Fold (OOF) predictions from all 8 base models are concatenated into a new feature matrix.
   * A **Ridge Regressor** (alpha=10.0) acts as the meta-model, learning the optimal weighting of the base models' predictions to output the final age.

This complete architecture is trained **independently** for 6 different cell-type configurations: `overall`, `CD4 T`, `CD8 T`, `NK`, `B cells`, and `monocytes`.

The results are found in the csv file `multi_celltype_ensemble_submission.csv`.
