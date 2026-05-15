# Aging Challenge 2026: Final Ensemble Pipeline

This directory contains the final evaluation and training pipeline for predicting chronological age using a combination of **scRNA-seq Pseudobulk**, **Geneformer embeddings**, and **scGPT embeddings**. 

The primary script (`models_script.py`) evaluates base learners on the **OneK1K** dataset, builds stacking ensembles using both OneK1K and **AIDA** datasets, and safely exports the fully trained models for future inference.

The code `extract_weights.py` extracts the ensemble weights from the saved .joblib models into a readable form and is saved in the results directory.

## 🚀 Pipeline Overview

The script is divided into three major operational steps:

### Step 1: Base Learner Evaluation & Feature Extraction
* Evaluates 8 individual base models exclusively on the **OneK1K** (Train + Val) dataset.
* Evaluates performance on the true OneK1K Test set.
* Extracts native feature weights for tree/linear models and calculates **Permutation Importance** for neural networks (MLP/KNN).
* Compiles a comprehensive top-to-bottom feature ranking across all modalities.

### Step 2: Cross-Dataset Ensemble Models
* Loads the **AIDA** dataset.
* **Feature Alignment:** Strictly aligns AIDA's gene symbols and embedding dimensions to the OneK1K structure using `.reindex()` to guarantee mathematical compatibility and prevent `NaN` generation.
* Uses 5-Fold Out-Of-Fold (OOF) predictions to train a **Ridge Meta-Model** over the 8 base learners.
* Builds and evaluates three distinct final ensembles:
  1. `Ensemble_OneK1K` (Trained strictly on OneK1K)
  2. `Ensemble_AIDA` (Trained strictly on AIDA)
  3. `Ensemble_Both` (Trained on combined OneK1K + AIDA)

### Step 4: Model Exporting (`saved_final_ensembles/`)
Saves the fully fitted pipelines, feature selectors, and meta-models to disk using `joblib` so they can be deployed without retraining.

---

## 📂 Expected Directory Structure
The script expects to be run from inside the `models/` folder and relies on the following relative paths at the project root:

```text
├── data/                            # OneK1K datasets
│   ├── metadata/
│   ├── scRNA-seq_pseudobulk/
│   ├── scRNA-seq_geneformer_pseudobulk/
│   └── scgpt_pseudobulk_tsv/        
├── data_AIDA/                       # AIDA datasets
│   ├── metadata/
│   ├── scRNA-seq_pseudobulk/
│   ├── gf_pseudobulk_tsv/
│   └── scgpt_pseudobulk_tsv/
├── results/                         # Generated metrics and rankings
│   └── true_test_ages.csv           # Required for test set evaluation
└── models/
    ├── models_script.py           # The main pipeline script
    ├── extract_weights.py        
    └── saved_final_ensembles/       # Output directory for serialized models
