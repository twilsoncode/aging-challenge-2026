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

def filter_by_cell_type(df, ct):
    """Filters dataframe columns based on cell type naming variants."""
    if ct == "All" or df is None or df.empty:
        return df
        
    selected = []
    for col in df.columns:
        col_lower = col.lower().replace('_', '').replace('-', '')
        if ct == "B_cells" and "bcell" in col_lower:
            selected.append(col)
        elif ct == "CD4_T" and "cd4t" in col_lower:
            selected.append(col)
        elif ct == "CD8_T" and "cd8t" in col_lower:
            selected.append(col)
        elif ct == "monocytes" and "monocyte" in col_lower:
            selected.append(col)
        elif ct == "NK" and "nk" in col_lower:
            selected.append(col)
            
    return df[selected]

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

# Define cell types to iterate over globally
cell_types_to_train = ["All", "B_cells", "CD4_T", "CD8_T", "monocytes", "NK"]

# ==========================================
# STEP 1: Evaluate Base Learners on OneK1K (All + 5 Cell Types)
# ==========================================
print("\n" + "="*50)
print("STEP 1: EVALUATING INDIVIDUAL LEARNERS ON ONEK1K (OVERALL & CELL TYPES)")
print("="*50)

step1_results = []
all_feature_rankings = []

for ct in cell_types_to_train:
    print(f"\n>>> PROCESSING BASE LEARNERS FOR: {ct} <<<")
    
    # 1. Filter data for the specific cell type
    X_pb_tv = filter_by_cell_type(pb_1k_trainval, ct)
    X_pb_test = filter_by_cell_type(test_1k["pb"], ct)
    X_gf_tv = filter_by_cell_type(gf_1k_trainval, ct)
    X_gf_test = filter_by_cell_type(test_1k["gf"], ct)
    X_scgpt_tv = filter_by_cell_type(scgpt_1k_trainval, ct)
    X_scgpt_test = filter_by_cell_type(test_1k["scgpt"], ct)

    # 2. Setup Pseudobulk Feature Selection
    if X_pb_tv.shape[1] > 0:
        k_best = min(2000, X_pb_tv.shape[1])
        try:
            pb_selector = Pipeline([
                ('variance', VarianceThreshold(threshold=0.01)),
                ('kbest', SelectKBest(score_func=f_regression, k=k_best))
            ])
            pb_tv_genes = pb_selector.fit_transform(X_pb_tv, y_1k_trainval)
            
            # Extract names by applying both masks sequentially
            var_mask = pb_selector.named_steps['variance'].get_support()
            kbest_mask = pb_selector.named_steps['kbest'].get_support()
            selected_genes = X_pb_tv.columns[var_mask][kbest_mask].tolist() + ['sex_binary']
            
        except ValueError:
            # Fallback if variance threshold fails (usually on tiny datasets)
            pb_selector = Pipeline([('kbest', SelectKBest(score_func=f_regression, k=k_best))])
            pb_tv_genes = pb_selector.fit_transform(X_pb_tv, y_1k_trainval)
            
            # Extract names using only the kbest mask
            kbest_mask = pb_selector.named_steps['kbest'].get_support()
            selected_genes = X_pb_tv.columns[kbest_mask].tolist() + ['sex_binary']
            
        pb_test_genes = pb_selector.transform(X_pb_test)
        
        pb_tv_sel = np.column_stack((pb_tv_genes, sex_1k_trainval.values))
        pb_test_sel = np.column_stack((pb_test_genes, test_1k["sex"].values))
    else:
        pb_tv_sel = sex_1k_trainval.values.reshape(-1, 1)
        pb_test_sel = test_1k["sex"].values.reshape(-1, 1)
        selected_genes = ['sex_binary']

    # 3. Setup Embedding Data
    gf_tv_sel = np.column_stack((X_gf_tv.values, sex_1k_trainval.values)) if X_gf_tv.shape[1] > 0 else sex_1k_trainval.values.reshape(-1, 1)
    gf_test_sel = np.column_stack((X_gf_test.values, test_1k["sex"].values)) if X_gf_test.shape[1] > 0 else test_1k["sex"].values.reshape(-1, 1)
    scgpt_tv_sel = np.column_stack((X_scgpt_tv.values, sex_1k_trainval.values)) if X_scgpt_tv.shape[1] > 0 else sex_1k_trainval.values.reshape(-1, 1)
    scgpt_test_sel = np.column_stack((X_scgpt_test.values, test_1k["sex"].values)) if X_scgpt_test.shape[1] > 0 else test_1k["sex"].values.reshape(-1, 1)

    # 4. Define Models
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

    # Evaluate Pseudobulk Models
    if X_pb_tv.shape[1] > 0:
        for name, model in tqdm(models_pb.items(), desc=f"PB Models ({ct})"):
            model.fit(pb_tv_sel, y_1k_trainval)
            
            if test_1k["y"] is not None:
                preds = model.predict(pb_test_sel)
                res = eval_metrics(test_1k["y"], preds, f"{name} (PB)", f"OneK1K Test ({ct})")
                res["Cell_Type"] = ct
                step1_results.append(res)
            
            if hasattr(model, 'feature_importances_'):
                imp = pd.Series(model.feature_importances_, index=selected_genes)
            elif hasattr(model, 'coef_'):
                imp = pd.Series(np.abs(model.coef_), index=selected_genes)
                
            df_imp = imp.sort_values(ascending=False).reset_index()
            df_imp.columns = ['Feature', 'Importance']
            df_imp['Model'] = f"{name} (PB)"
            df_imp['Cell_Type'] = ct
            df_imp['Rank'] = np.arange(1, len(df_imp) + 1)
            all_feature_rankings.append(df_imp)

    # Evaluate Embeddings Models
    emb_datasets = []
    if X_gf_tv.shape[1] > 0:
        emb_datasets.append(("Geneformer", gf_tv_sel, gf_test_sel, [f"GF_dim_{i}" for i in range(X_gf_tv.shape[1])] + ['sex_binary']))
    if X_scgpt_tv.shape[1] > 0:
        emb_datasets.append(("scGPT", scgpt_tv_sel, scgpt_test_sel, [f"scGPT_dim_{i}" for i in range(X_scgpt_tv.shape[1])] + ['sex_binary']))

    for emb_name, tv_data, test_data, feat_names in emb_datasets:
        for name, model in tqdm(models_emb.items(), desc=f"{emb_name} Models ({ct})"):
            model.fit(tv_data, y_1k_trainval)
            
            if test_1k["y"] is not None:
                preds = model.predict(test_data)
                res = eval_metrics(test_1k["y"], preds, f"{name} ({emb_name})", f"OneK1K Test ({ct})")
                res["Cell_Type"] = ct
                step1_results.append(res)
                perm_X, perm_y = test_data, test_1k["y"]
            else:
                perm_X, perm_y = tv_data, y_1k_trainval
                
            perm_importance = permutation_importance(model, perm_X, perm_y, n_repeats=5, random_state=42, n_jobs=-1)
            imp_sorted = pd.Series(perm_importance.importances_mean, index=feat_names).sort_values(ascending=False)
            
            df_imp = imp_sorted.reset_index()
            df_imp.columns = ['Feature', 'Importance']
            df_imp['Model'] = f"{name} ({emb_name})"
            df_imp['Cell_Type'] = ct
            df_imp['Rank'] = np.arange(1, len(df_imp) + 1)
            all_feature_rankings.append(df_imp)

# Save Step 1 Base Learner Metrics
pd.DataFrame(step1_results).to_csv(RESULTS_DIR / "onek1k_base_learner_metrics.csv", index=False)
pd.concat(all_feature_rankings, ignore_index=True).to_csv(RESULTS_DIR / "onek1k_base_learner_feature_importances.csv", index=False)
print(f"\n[Success] OneK1K Base Learner metrics & importances saved to {RESULTS_DIR}")

# ==========================================
# STEP 2: Rebuild Final Ensembles (Iterating Over Cell Types & Subsets)
# ==========================================
print("\n" + "="*50)
print("STEP 2: ENSEMBLE CROSS-DATASET EVALUATION")
print("="*50)

print("Loading AIDA Data...")
train_aida = load_dataset("aida", "train")
val_aida = load_dataset("aida", "val")
test_aida = load_dataset("aida", "test")

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

def train_ensemble(X_pb, X_gf, X_scgpt, sex, y, desc="Ensemble", modalities=('pb', 'gf', 'scgpt')):
    if 'pb' in modalities and X_pb.shape[1] > 0:
        k_best = min(2000, X_pb.shape[1])
        try:
            selector = Pipeline([
                ('variance', VarianceThreshold(threshold=0.01)),
                ('kbest', SelectKBest(score_func=f_regression, k=k_best))
            ])
            pb_genes = selector.fit_transform(X_pb, y)
        except ValueError:
            selector = Pipeline([('kbest', SelectKBest(score_func=f_regression, k=k_best))])
            pb_genes = selector.fit_transform(X_pb, y)
    else:
        selector = None
        pb_genes = np.empty((len(y), 0))

    X_pb_sel = np.column_stack((pb_genes, sex.values)) if pb_genes.shape[1] > 0 else sex.values.reshape(-1, 1)
    X_gf_sel = np.column_stack((X_gf.values, sex.values)) if X_gf.shape[1] > 0 else sex.values.reshape(-1, 1)
    X_scgpt_sel = np.column_stack((X_scgpt.values, sex.values)) if X_scgpt.shape[1] > 0 else sex.values.reshape(-1, 1)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models_def = []
    
    if 'pb' in modalities and X_pb.shape[1] > 0:
        models_def.extend([
            (XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1), X_pb_sel, "PB_XGB"),
            (HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42), X_pb_sel, "PB_LGBM"),
            (RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42), X_pb_sel, "PB_RF"),
            (ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42), X_pb_sel, "PB_ElasticNet")
        ])
    if 'gf' in modalities and X_gf.shape[1] > 0:
        models_def.extend([
            (Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(256, 64), random_state=42))]), X_gf_sel, "GF_MLP"),
            (Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=15, weights='distance'))]), X_gf_sel, "GF_KNN")
        ])
    if 'scgpt' in modalities and X_scgpt.shape[1] > 0:
        models_def.extend([
            (Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(256, 64), random_state=42))]), X_scgpt_sel, "scGPT_MLP"),
            (Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=15, weights='distance'))]), X_scgpt_sel, "scGPT_KNN")
        ])

    if not models_def:
        return None 

    oof_preds = np.zeros((len(y), len(models_def)))
    y_arr = y.values
    
    for i, (model, X, name) in enumerate(tqdm(models_def, desc=f"Training {desc}")):
        for tr_idx, va_idx in cv.split(X):
            model.fit(X[tr_idx], y_arr[tr_idx])
            oof_preds[va_idx, i] = model.predict(X[va_idx])
            
    meta = Ridge(alpha=10.0, random_state=42)
    meta.fit(oof_preds, y_arr)
    
    trained_models = []
    for model, X, name in models_def:
        model.fit(X, y_arr)
        trained_models.append((model, name.split('_')[0].lower(), name))
        
    return trained_models, meta, [name for _, _, name in trained_models], selector

def predict_ensemble(models_tuple, X_pb, X_gf, X_scgpt, sex):
    base_models, meta, _, selector = models_tuple
    
    if selector is not None and X_pb.shape[1] > 0:
        pb_genes = selector.transform(X_pb)
    else:
        pb_genes = np.empty((len(sex), 0))
        
    features = {
        'pb': np.column_stack((pb_genes, sex.values)) if pb_genes.shape[1] > 0 else sex.values.reshape(-1, 1),
        'gf': np.column_stack((X_gf.values, sex.values)) if X_gf.shape[1] > 0 else sex.values.reshape(-1, 1),
        'scgpt': np.column_stack((X_scgpt.values, sex.values)) if X_scgpt.shape[1] > 0 else sex.values.reshape(-1, 1)
    }
    
    preds = np.column_stack([model.predict(features[mod_type]) for model, mod_type, name in base_models])
    return meta.predict(preds)

# The combinations that automatically create your 3 base datasets * 3 modality sets (including the extra 6 permutations)
datasets = [
    ("OneK1K", pb_1k_trainval, gf_1k_trainval, scgpt_1k_trainval, sex_1k_trainval, y_1k_trainval),
    ("AIDA", pb_aida_trainval, gf_aida_trainval, scgpt_aida_trainval, sex_aida_trainval, y_aida_trainval),
    ("Both", pb_both_trainval, gf_both_trainval, scgpt_both_trainval, sex_both_trainval, y_both_trainval)
]

modalities = [
    ("All_Mods", ('pb', 'gf', 'scgpt')),
    ("PB_GF", ('pb', 'gf')),
    ("PB_scGPT", ('pb', 'scgpt'))
]

all_ensembles = []

print("\n--- Building Ensembles (Base Models Skipped for Speed) ---")
for ct in cell_types_to_train:
    print(f"\n>>> PROCESSING ENSEMBLES FOR: {ct} <<<")
    for ds_name, X_pb, X_gf, X_scgpt, sex, y in datasets:
        X_pb_ct = filter_by_cell_type(X_pb, ct)
        X_gf_ct = filter_by_cell_type(X_gf, ct)
        X_scgpt_ct = filter_by_cell_type(X_scgpt, ct)
        
        for mod_name, mods in modalities:
            ens_name = f"{ds_name}_{mod_name}_{ct}"
            models_tuple = train_ensemble(X_pb_ct, X_gf_ct, X_scgpt_ct, sex, y, ens_name, mods)
            if models_tuple is not None:
                all_ensembles.append((ens_name, ct, models_tuple))
            else:
                print(f"[Skipped] {ens_name} - Insufficient features.")

# Extract and save Reliance Weights
weights_data = []
for ens_name, ct, (_, meta, model_names, _) in all_ensembles:
    for model_name, weight in zip(model_names, meta.coef_):
        weights_data.append({"Ensemble": ens_name, "Cell_Type": ct, "Base_Model": model_name, "Reliance_Weight": weight})

pd.DataFrame(weights_data).to_csv(RESULTS_DIR / "ensemble_reliance_weights.csv", index=False)

# --- Evaluate All Ensembles ---
final_results = []
test_sets = [
    ("OneK1K Test", test_1k["pb"], test_1k["gf"], test_1k["scgpt"], test_1k["sex"], test_1k["y"]),
    ("AIDA Test", test_aida["pb"], test_aida["gf"], test_aida["scgpt"], test_aida["sex"], test_aida["y"])
]

print("\n--- Final Cross-Dataset Evaluation ---")
for model_name, ct, model_obj in all_ensembles:
    for test_name, X_pb_test, X_gf_test, X_scgpt_test, sex_test, y_true in test_sets:
        if y_true is not None:
            X_pb_test_ct = filter_by_cell_type(X_pb_test, ct)
            X_gf_test_ct = filter_by_cell_type(X_gf_test, ct)
            X_scgpt_test_ct = filter_by_cell_type(X_scgpt_test, ct)
            
            try:
                preds = predict_ensemble(model_obj, X_pb_test_ct, X_gf_test_ct, X_scgpt_test_ct, sex_test)
                res = eval_metrics(y_true, preds, model_name, test_name)
                print(f"[{test_name}] {model_name} -> MAE: {res['MAE']:.4f} | R2: {res['R2']:.4f} | Pearson: {res['Pearson']:.4f}")
                final_results.append(res)
            except Exception as e:
                print(f"[Error] Failed evaluating {model_name} on {test_name}: {e}")

if final_results:
    pd.DataFrame(final_results).to_csv(RESULTS_DIR / "final_ensemble_evaluation_report.csv", index=False)

# ==========================================
# STEP 4: Save Final Models for Later Use
# ==========================================
print("\n" + "="*50)
print("STEP 4: SAVING MODELS TO DISK")
print("="*50)

def save_ensemble_to_disk(models_tuple, prefix):
    base_models, meta, model_names, selector = models_tuple
    sub_dir = MODELS_DIR / prefix
    sub_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(meta, sub_dir / f"{prefix}_meta_ridge.joblib")
    if selector is not None:
        joblib.dump(selector, sub_dir / f"{prefix}_pb_selector.joblib")
    
    for (model, mod_type, name) in base_models:
        joblib.dump(model, sub_dir / f"{prefix}_base_{name}.joblib")

for model_name, ct, model_obj in tqdm(all_ensembles, desc="Saving Ensembles"):
    save_ensemble_to_disk(model_obj, model_name)

print(f"\nAll operations complete.")
