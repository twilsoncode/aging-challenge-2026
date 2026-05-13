# Models — Age Prediction Baseline

## Scripts

| Script | Purpose |
|--------|---------|
| `train_age_model.py` | Train a Random Forest and generate val/test predictions |
| `evaluate_val.py` | Score your validation predictions (competitor use) |
| `further_models.py` | The winning ensemble model code |

## Saved model joblib files and final feature summaries
In the folder `saved_ensembles_6_types` you will find the saved base learners and meta-learners for each of the 6 models (one overall and five for each cell type). This can be loaded so that the `further_models.py` script does not need to be run each time. There are also reports of the top features in each of the six models for further investigation. Using 32 cores on Iridis, the `further_models.py` workflow took ~ 1.5 hours.

## Quick start

```bash
# 1. Train baseline (place competition h5ad under data/ — see data/README.md)
python models/train_age_model.py \
    --input data/scRNA-seq_pseudobulk/train_pseudobulk_donor_aggregated_public.h5ad

# 2. Evaluate on validation set
python models/evaluate_val.py --plot
```

The teaching notebooks write runs under `results/` by passing `--output-dir`. If you omit `--input`, defaults assume the **`data/`** layout from `data/README.md` (shared scratch) and `models/output/` for runs.

## train_age_model.py — options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | auto-discovered | Donor-aggregated pseudobulk h5ad |
| `--n-genes` | 2000 | Top genes by variance |
| `--all-features` | off | Use all features (no gene selection) |
| `--n-estimators` | 200 | Number of trees |
| `--max-depth` | None | Max tree depth (None = unlimited) |
| `--seed` | 42 | Random seed |
| `--sex` | off | Add donor sex as binary feature |
| `--donor-metadata` | auto | Path to donor_metadata.csv (needed for `--sex`) |
| `--geneformer` | off | Append Geneformer embeddings (see notebook 04) |
| `--geneformer-only` | off | Use Geneformer as sole feature set |
| `--compare-pca` | off | Run multiple configurations and compare |
| `--output-dir` | auto-timestamped | Where to save results |

## Outputs (models/output/TIMESTAMP/)

| File | Description |
|------|-------------|
| `*_rf_model.joblib` | Saved Random Forest model |
| `*_feature_names.csv` | Ordered list of features used |
| `*_top_features.csv` | Top 20 features by importance |
| `*_test_predictions.csv` | Predictions for test donors (submit this) |
| `val_predictions.csv` | Predictions for val donors (self-evaluate) |
| `test_predictions.csv` | Copy of the best config's test predictions |

## Submission file

Take `test_predictions.csv` and rename the prediction column to match the submission format (`donor_id`, `age`):

```python
import pandas as pd
df = pd.read_csv("models/output/TIMESTAMP/test_predictions.csv")
df = df.rename(columns={"predicted_age": "age"})
df[["donor_id", "age"]].to_csv("my_submission.csv", index=False)
```
