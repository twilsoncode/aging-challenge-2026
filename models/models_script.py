import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
import joblib
from tqdm.auto import tqdm

from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 0. Setup Paths & Helper Functions
# ==========================================
PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_ONE = PROJ_ROOT / "data"
DATA_DIR_AIDA = PROJ_ROOT / "data_AIDA"
RESULTS_DIR = PROJ_ROOT / "results"
MODELS_DIR = PROJ_ROOT / "models" / "saved_final_ensembles"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def eval_metrics(y_true, y_pred, model_name, test_set_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    p_corr, _ = pearsonr(y_true, y_pred)
    s_corr, _ = spearmanr(y_true, y_pred)
    
    return {"Model": model_name, "Test_Set": test_set_name, "MAE": mae, "RMSE": rmse, "R2": r2, "Pearson": p_corr, "Spearman": s_corr}

# ==========================================
# 1. Data Loading Functions
# ==========================================
def load_dataset(dataset_name="onek1k", split="train"):
    """Loads target, pseudobulk, geneformer, and scgpt data for a given split and dataset."""
    is_aida = (dataset_name == "aida")
    base_dir = DATA_DIR_AIDA if is_aida else DATA_DIR_ONE
    
    # Load Targets & Metadata
    if is_aida:
        meta = pd.read_csv(base_dir / "metadata/donor_metadata.csv").set_index("donor_id")
        y = meta['age'].dropna()
        sex = meta[['sex_binary']].fillna(0)
    else:
        meta = pd.read_csv(base_dir / "metadata/donor_metadata.csv").set_index("donor_id")
        
        if split == "test":
            age_file = RESULTS_DIR / "true_test_ages.csv"
        else:
            age_file = base_dir / f"metadata/{split}_age.csv"
            
        if age_file.exists():
            y = pd.read_csv(age_file).set_index("donor_id")["age"]
            sex = meta.loc[y.index, ['sex_binary']].fillna(0)
        else:
            y = None
            sex = meta[['sex_binary']].fillna(0)

    # Load Pseudobulk
    pb_suffix = "_public" if not is_aida else ""
    adata = sc.read_h5ad(base_dir / f"scRNA-seq_pseudobulk/{split}_pseudobulk_donor_aggregated{pb_suffix}.h5ad")
    X_pb = adata.X.toarray() if issparse(adata.X) else adata.X
    pb_df = pd.DataFrame(np.log1p(X_pb), index=adata.obs["donor_id"].astype(int), columns=adata.var_names)

    # Load Geneformer
    gf_prefix = "geneformer_aida" if is_aida else "geneformer"
    gf_dir = "gf_pseudobulk_tsv" if is_aida else "scRNA-seq_geneformer_pseudobulk"
    gf_df = pd.read_csv(base_dir / f"{gf_dir}/{gf_prefix}_pseudobulk_{split}.tsv.gz", sep='\t').set_index("donor_id").filter(regex='__emb')

    # Load scGPT
    # UPDATED: Pointing to the new directory for OneK1K scGPT data
    if is_aida:
        scgpt_prefix = "scgpt_aida"
        scgpt_dir = "scgpt_pseudobulk_tsv"
    else:
        scgpt_prefix = "scgpt"
        scgpt_dir = "scgpt_pseudobulk_tsv"
        
    try:
        scgpt_df = pd.read_csv(base_dir / f"{scgpt_dir}/{scgpt_prefix}_pseudobulk_{split}.tsv.gz", sep='\t').set_index("donor_id").filter(regex='__emb')
    except FileNotFoundError:
        scgpt_df = pd.DataFrame(0, index=gf_df.index, columns=[f"scgpt_emb_{i}" for i in range(512)])

    # Align indices
    valid_idx = pb_df.index
    if y is not None:
        valid_idx = valid_idx.intersection(y.index)
    
    return {
        "pb": pb_df.loc[valid_idx].fillna(0),
        "gf": gf_df.loc[valid_idx].fillna(0),
        "scgpt": scgpt_df.loc[valid_idx].fillna(0),
        "sex": sex.loc[valid_idx],
        "y": y.loc[valid_idx] if y is not None else None
    }

print("Loading OneK1K Data...")
train_1k = load_dataset("onek1k", "train")
val_1k = load_dataset("onek1k", "val")
test_1k = load_dataset("onek1k", "test")

pb_1k_trainval = pd.concat([train_1k["pb"], val_1k["pb"]])
gf_1k_trainval = pd.concat([train_1k["gf"], val_1k["gf"]])
scgpt_1k_trainval = pd.concat([train_1k["scgpt"], val_1k["scgpt"]])
sex_1k_trainval = pd.concat([train_1k["sex"], val_1k["sex"]])
y_1k_trainval = pd.concat([train_1k["y"], val_1k["y"]])

# ==========================================
# STEP 1: Separate Learners Evaluation (OneK1K)
# ==========================================
print("\n" + "="*50)
print("STEP 1: EVALUATING INDIVIDUAL LEARNERS & EXTRACTING ALL FEATURES")
print("="*50)

pb_selector = Pipeline([
    ('variance', VarianceThreshold(threshold=0.01)),
    ('kbest', SelectKBest(score_func=f_regression, k=2000)) 
])
pb_1k_tv_sel = np.column_stack((pb_selector.fit_transform(pb_1k_trainval, y_1k_trainval), sex_1k_trainval.values))
pb_1k_test_sel = np.column_stack((pb_selector.transform(test_1k["pb"]), test_1k["sex"].values))

gf_1k_tv_sel = np.column_stack((gf_1k_trainval.values, sex_1k_trainval.values))
gf_1k_test_sel = np.column_stack((test_1k["gf"].values, test_1k["sex"].values))

scgpt_1k_tv_sel = np.column_stack((scgpt_1k_trainval.values, sex_1k_trainval.values))
scgpt_1k_test_sel = np.column_stack((test_1k["scgpt"].values, test_1k["sex"].values))

models_pb = {
    "XGB": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1),
    "LGBM": HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42),
    "RF": RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
}

models_emb = {
    "MLP": Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(256, 64), alpha=0.1, random_state=42))]),
    "KNN": Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1))])
}

step1_results = []
all_feature_rankings = []
gene_cols = pb_1k_trainval.columns
var_mask = pb_selector.named_steps['variance'].get_support()
kbest_mask = pb_selector.named_steps['kbest'].get_support()
selected_genes = gene_cols[var_mask][kbest_mask].tolist() + ['sex_binary']

# Evaluate Pseudobulk Models
print("\n--- Training Pseudobulk Models ---")
for name, model in tqdm(models_pb.items(), desc="PB Models"):
    model.fit(pb_1k_tv_sel, y_1k_trainval)
    
    if test_1k["y"] is not None:
        preds = model.predict(pb_1k_test_sel)
        step1_results.append(eval_metrics(test_1k["y"], preds, f"{name} (Pseudobulk)", "OneK1K Test"))
    
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=selected_genes)
    elif hasattr(model, 'coef_'):
        imp = pd.Series(np.abs(model.coef_), index=selected_genes)
        
    imp_sorted = imp.sort_values(ascending=False)
    df_imp = imp_sorted.reset_index()
    df_imp.columns = ['Feature', 'Importance']
    df_imp['Model'] = f"{name} (Pseudobulk)"
    df_imp['Rank'] = np.arange(1, len(df_imp) + 1)
    all_feature_rankings.append(df_imp)

# Evaluate Embeddings
print("\n--- Training Embedding Models (with Permutation Importance) ---")
emb_datasets = [("Geneformer", gf_1k_tv_sel, gf_1k_test_sel), ("scGPT", scgpt_1k_tv_sel, scgpt_1k_test_sel)]

for emb_name, tv_data, test_data in emb_datasets:
    emb_dim = tv_data.shape[1] - 1 
    emb_feature_names = [f"{emb_name}_dim_{i}" for i in range(emb_dim)] + ['sex_binary']
    
    for name, model in tqdm(models_emb.items(), desc=f"{emb_name} Models"):
        model.fit(tv_data, y_1k_trainval)
        
        if test_1k["y"] is not None:
            preds = model.predict(test_data)
            step1_results.append(eval_metrics(test_1k["y"], preds, f"{name} ({emb_name})", "OneK1K Test"))
            perm_X, perm_y = test_data, test_1k["y"]
        else:
            perm_X, perm_y = tv_data, y_1k_trainval
            
        perm_importance = permutation_importance(model, perm_X, perm_y, n_repeats=5, random_state=42, n_jobs=-1)
        imp_sorted = pd.Series(perm_importance.importances_mean, index=emb_feature_names).sort_values(ascending=False)
        
        df_imp = imp_sorted.reset_index()
        df_imp.columns = ['Feature', 'Importance']
        df_imp['Model'] = f"{name} ({emb_name})"
        df_imp['Rank'] = np.arange(1, len(df_imp) + 1)
        all_feature_rankings.append(df_imp)

ranking_out_path = RESULTS_DIR / "full_feature_rankings_step1.csv"
pd.concat(all_feature_rankings, ignore_index=True).to_csv(ranking_out_path, index=False)
print(f"\n[Success] Feature rankings saved to: {ranking_out_path}")

# ==========================================
# STEP 2: Rebuild Final Models (OneK1K, AIDA, Both)
# ==========================================
print("\n" + "="*50)
print("STEP 2: ENSEMBLE CROSS-DATASET EVALUATION")
print("="*50)

print("Loading AIDA Data...")
train_aida = load_dataset("aida", "train")
val_aida = load_dataset("aida", "val")
test_aida = load_dataset("aida", "test")

# REQUIRED: Align AIDA columns to OneK1K to prevent NaNs during concatenation
pb_aida_trainval = pd.concat([train_aida["pb"], val_aida["pb"]]).reindex(columns=pb_1k_trainval.columns, fill_value=0)
test_aida["pb"] = test_aida["pb"].reindex(columns=pb_1k_trainval.columns, fill_value=0)

gf_aida_trainval = pd.concat([train_aida["gf"], val_aida["gf"]]).reindex(columns=gf_1k_trainval.columns, fill_value=0)
test_aida["gf"] = test_aida["gf"].reindex(columns=gf_1k_trainval.columns, fill_value=0)

scgpt_aida_trainval = pd.concat([train_aida["scgpt"], val_aida["scgpt"]]).reindex(columns=scgpt_1k_trainval.columns, fill_value=0)
test_aida["scgpt"] = test_aida["scgpt"].reindex(columns=scgpt_1k_trainval.columns, fill_value=0)

sex_aida_trainval = pd.concat([train_aida["sex"], val_aida["sex"]])
y_aida_trainval = pd.concat([train_aida["y"], val_aida["y"]])

pb_both_trainval = pd.concat([pb_1k_trainval, pb_aida_trainval])
gf_both_trainval = pd.concat([gf_1k_trainval, gf_aida_trainval])
scgpt_both_trainval = pd.concat([scgpt_1k_trainval, scgpt_aida_trainval])
sex_both_trainval = pd.concat([sex_1k_trainval, sex_aida_trainval])
y_both_trainval = pd.concat([y_1k_trainval, y_aida_trainval])

def train_ensemble(X_pb, X_gf, X_scgpt, sex, y, desc="Ensemble"):
    pb_genes = pb_selector.transform(X_pb)
    X_pb_sel = np.column_stack((pb_genes, sex.values))
    X_gf_sel = np.column_stack((X_gf.values, sex.values))
    X_scgpt_sel = np.column_stack((X_scgpt.values, sex.values))
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(y), 8))
    
    models = [
        (XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1), X_pb_sel),
        (HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42), X_pb_sel),
        (RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42), X_pb_sel),
        (ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42), X_pb_sel),
        (Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(256, 64), random_state=42))]), X_gf_sel),
        (Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=15, weights='distance'))]), X_gf_sel),
        (Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(256, 64), random_state=42))]), X_scgpt_sel),
        (Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=15, weights='distance'))]), X_scgpt_sel)
    ]
    
    y_arr = y.values
    for i, (model, X) in enumerate(tqdm(models, desc=f"Training {desc} (OOF Folds)")):
        for tr_idx, va_idx in cv.split(X):
            model.fit(X[tr_idx], y_arr[tr_idx])
            oof_preds[va_idx, i] = model.predict(X[va_idx])
            
    meta = Ridge(alpha=10.0, random_state=42)
    meta.fit(oof_preds, y_arr)
    
    for model, X in models:
        model.fit(X, y_arr)
        
    return models, meta

def predict_ensemble(models_and_meta, X_pb, X_gf, X_scgpt, sex):
    base_models, meta = models_and_meta
    pb_genes = pb_selector.transform(X_pb)
    X_pb_sel = np.column_stack((pb_genes, sex.values))
    X_gf_sel = np.column_stack((X_gf.values, sex.values))
    X_scgpt_sel = np.column_stack((X_scgpt.values, sex.values))
    
    test_features = [X_pb_sel]*4 + [X_gf_sel]*2 + [X_scgpt_sel]*2
    preds = np.column_stack([model.predict(X) for (model, _), X in zip(base_models, test_features)])
    return meta.predict(preds)

print("\n--- Building Final Ensembles ---")
ens_onek1k = train_ensemble(pb_1k_trainval, gf_1k_trainval, scgpt_1k_trainval, sex_1k_trainval, y_1k_trainval, "Model 1: OneK1K")
ens_aida = train_ensemble(pb_aida_trainval, gf_aida_trainval, scgpt_aida_trainval, sex_aida_trainval, y_aida_trainval, "Model 2: AIDA")
ens_both = train_ensemble(pb_both_trainval, gf_both_trainval, scgpt_both_trainval, sex_both_trainval, y_both_trainval, "Model 3: Combined")

# --- Evaluate 3 Models ---
final_results = []
test_sets = [
    ("OneK1K Test", test_1k["pb"], test_1k["gf"], test_1k["scgpt"], test_1k["sex"], test_1k["y"]),
    ("AIDA Test", test_aida["pb"], test_aida["gf"], test_aida["scgpt"], test_aida["sex"], test_aida["y"])
]

print("\n--- Final Cross-Dataset Evaluation ---")
for model_name, model_obj in [("Ensemble_OneK1K", ens_onek1k), 
                              ("Ensemble_AIDA", ens_aida), 
                              ("Ensemble_Both", ens_both)]:
    for test_name, X_pb, X_gf, X_scgpt, sex, y_true in test_sets:
        if y_true is not None:
            preds = predict_ensemble(model_obj, X_pb, X_gf, X_scgpt, sex)
            res = eval_metrics(y_true, preds, model_name, test_name)
            print(f"[{test_name}] {model_name}")
            print(f"  MAE: {res['MAE']:.4f} | RMSE: {res['RMSE']:.4f} | R2: {res['R2']:.4f} | Pearson: {res['Pearson']:.4f} | Spearman: {res['Spearman']:.4f}\n")
            final_results.append(res)
        else:
            print(f"[{test_name}] {model_name}: Skipping metrics (No true labels available)")

if step1_results or final_results:
    all_results_df = pd.DataFrame(step1_results + final_results)
    report_path = RESULTS_DIR / "final_model_evaluation_report.csv"
    all_results_df.to_csv(report_path, index=False)

# ==========================================
# STEP 4: Save Final Models for Later Use
# ==========================================
print("\n" + "="*50)
print("STEP 4: SAVING MODELS TO DISK")
print("="*50)

joblib.dump(pb_selector, MODELS_DIR / "master_pb_selector.joblib")
print(f"Saved global feature selector to {MODELS_DIR}/master_pb_selector.joblib")

def save_ensemble_to_disk(models_and_meta, prefix):
    base_models, meta = models_and_meta
    sub_dir = MODELS_DIR / prefix
    sub_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(meta, sub_dir / f"{prefix}_meta_ridge.joblib")
    model_names = ["XGB", "LGBM", "RF", "ElasticNet", "GF_MLP", "GF_KNN", "scGPT_MLP", "scGPT_KNN"]
    
    for (model, _), name in zip(base_models, model_names):
        joblib.dump(model, sub_dir / f"{prefix}_base_{name}.joblib")
        
    print(f"Saved {prefix} ensemble to {sub_dir}/")

save_ensemble_to_disk(ens_onek1k, "Ensemble_OneK1K")
save_ensemble_to_disk(ens_aida, "Ensemble_AIDA")
save_ensemble_to_disk(ens_both, "Ensemble_Both")

print(f"\nAll operations complete.")