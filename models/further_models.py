import time
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import scanpy as sc
import joblib

from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse

# Suppress warnings to keep the terminal clean
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 1. Setup Paths
# ==========================================
PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_ROOT / "data"
RESULTS_DIR = PROJ_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = PROJ_ROOT / "models" / "saved_ensembles_6_types"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Starting MULTI-CELL-TYPE Ensemble Pipeline...")

# ==========================================
# 2. Load Targets & Metadata (Combined)
# ==========================================
print("Loading and combining training & validation labels...")
y_train = pd.read_csv(DATA_DIR / "metadata/train_age.csv").set_index("donor_id")["age"]
y_val = pd.read_csv(DATA_DIR / "metadata/val_age.csv").set_index("donor_id")["age"]
y_full = pd.concat([y_train, y_val])

meta = pd.read_csv(DATA_DIR / "metadata/donor_metadata.csv").set_index("donor_id")
sex_train = meta.loc[y_train.index, ['sex_binary']].fillna(0)
sex_val = meta.loc[y_val.index, ['sex_binary']].fillna(0)
sex_full = pd.concat([sex_train, sex_val])

# Ensure test metadata is available
test_pb_path = DATA_DIR / "scRNA-seq_pseudobulk/test_pseudobulk_donor_aggregated_public.h5ad"
test_data_available = test_pb_path.exists()

# ==========================================
# 3. Dynamic Data Loaders
# ==========================================
def load_pseudobulk(split, cell_type="overall"):
    adata = sc.read_h5ad(DATA_DIR / f"scRNA-seq_pseudobulk/{split}_pseudobulk_donor_aggregated_public.h5ad")
    
    # Extract matrix and format to DataFrame
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    df = pd.DataFrame(X, index=adata.obs.index, columns=adata.var_names)
    df['donor_id'] = adata.obs['donor_id'].astype(int).values
    
    # Filter by cell type if requested and available in .obs
    if cell_type != "overall" and 'cell_type' in adata.obs.columns:
        df = df[adata.obs['cell_type'] == cell_type]
        
    # Aggregate to donor level (sum counts) to handle multiple rows per donor
    df_donor = df.groupby('donor_id').sum()
    
    return pd.DataFrame(np.log1p(df_donor), index=df_donor.index, columns=df_donor.columns)

def load_geneformer(split, cell_type="overall"):
    df = pd.read_csv(DATA_DIR / f"scRNA-seq_geneformer_pseudobulk/geneformer_pseudobulk_{split}.tsv.gz", sep='\t')
    df = df.set_index("donor_id")
    
    if cell_type == "overall":
        return df.filter(regex='__emb')
    else:
        # Filter columns belonging specifically to the requested cell type
        ct_safe = cell_type.replace(' ', '_')
        cols = [c for c in df.columns if ct_safe in c or cell_type in c]
        if not cols: 
            # Fallback if no specific columns found, return generic embeddings
            return df.filter(regex='__emb')
        return df[cols].filter(regex='__emb')

def load_genotype_pcs(split):
    df = pd.read_csv(DATA_DIR / f"genotypes/pca_{split}.tsv", sep='\t').set_index("donor_id")
    return df.select_dtypes(include=['number'])

# ==========================================
# 4. Core Pipeline Engine
# ==========================================
def train_and_predict_pipeline(model_name, pb_tr, gf_tr, pc_tr, sex_tr, y_tr, 
                               pb_te, gf_te, pc_te, sex_te):
    """Encapsulates the entire feature selection, 5-fold CV, prediction, saving, and interpretability logic."""
    print(f"\n--- Running Pipeline for: {model_name.upper()} ---")
    
    # 1. Feature Selection
    pb_selector = Pipeline([
        ('variance', VarianceThreshold(threshold=0.01)),
        ('kbest', SelectKBest(score_func=f_regression, k='all')) 
    ])
    
    pb_tr_genes = pb_selector.fit_transform(pb_tr, y_tr)
    pb_tr_sel = np.column_stack((pb_tr_genes, sex_tr.values))
    gf_tr_sel = np.column_stack((gf_tr.values, sex_tr.values))
    pc_tr_sel = np.column_stack((pc_tr.values, sex_tr.values))

    if test_data_available:
        pb_te_sel = np.column_stack((pb_selector.transform(pb_te), sex_te.values))
        gf_te_sel = np.column_stack((gf_te.values, sex_te.values))
        pc_te_sel = np.column_stack((pc_te.values, sex_te.values))
        
    # 2. Base Models Configuration
    m_xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1)
    m_lgb = HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42)
    m_rf = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    m_en = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    m_mlp = Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(256, 64), alpha=0.1, max_iter=400, random_state=42))])
    m_knn = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1))])
    m_ridge = Ridge(alpha=1.0, random_state=42)
    m_svr = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf', C=10.0, epsilon=0.1))])

    # 3. Generate OOF Predictions
    cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    def get_oof(model, X, y):
        oof = np.zeros(len(y))
        X_arr = X if isinstance(X, np.ndarray) else X.values
        y_arr = y if isinstance(y, np.ndarray) else y.values
        for train_idx, val_idx in cv_kfold.split(X_arr):
            model.fit(X_arr[train_idx], y_arr[train_idx])
            oof[val_idx] = model.predict(X_arr[val_idx])
        return oof

    print("  -> Generating 5-Fold OOF predictions...")
    oof_xgb = get_oof(m_xgb, pb_tr_sel, y_tr)
    oof_lgb = get_oof(m_lgb, pb_tr_sel, y_tr)
    oof_rf  = get_oof(m_rf, pb_tr_sel, y_tr)
    oof_en  = get_oof(m_en, pb_tr_sel, y_tr)
    oof_mlp = get_oof(m_mlp, gf_tr_sel, y_tr)
    oof_knn = get_oof(m_knn, gf_tr_sel, y_tr)
    oof_ridge = get_oof(m_ridge, pc_tr_sel, y_tr)
    oof_svr   = get_oof(m_svr, pc_tr_sel, y_tr)

    # 4. Meta-Model Evaluation & Training
    X_meta_tr = np.column_stack((oof_xgb, oof_lgb, oof_rf, oof_en, oof_mlp, oof_knn, oof_ridge, oof_svr))
    meta_model = Ridge(alpha=10.0, random_state=42)
    
    # Optional: print 1 overall CV metric for this specific cell type model
    oofs_meta = get_oof(Ridge(alpha=10.0, random_state=42), X_meta_tr, y_tr)
    print(f"  -> [{model_name}] OOF Meta-Model MAE: {mean_absolute_error(y_tr, oofs_meta):.4f}")
    meta_model.fit(X_meta_tr, y_tr)

    # 5. Retrain Base Models on 100% Data
    print("  -> Refitting base models on 100% of data...")
    m_xgb.fit(pb_tr_sel, y_tr)
    m_lgb.fit(pb_tr_sel, y_tr)
    m_rf.fit(pb_tr_sel, y_tr)
    m_en.fit(pb_tr_sel, y_tr)
    m_mlp.fit(gf_tr_sel, y_tr)
    m_knn.fit(gf_tr_sel, y_tr)
    m_ridge.fit(pc_tr_sel, y_tr)
    m_svr.fit(pc_tr_sel, y_tr)

    # ==========================================
    # 6. Save Models for this specific Cell Type
    # ==========================================
    # Create a sub-folder for this specific cell type (e.g., 'CD4_T')
    cell_model_dir = MODEL_DIR / model_name.replace(" ", "_")
    cell_model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  -> Saving all {model_name} models to {cell_model_dir}...")
    joblib.dump(pb_selector, cell_model_dir / 'pb_selector.joblib')
    joblib.dump(m_xgb, cell_model_dir / 'model_xgb.joblib')
    joblib.dump(m_lgb, cell_model_dir / 'model_lgb.joblib')
    joblib.dump(m_rf, cell_model_dir / 'model_rf.joblib')
    joblib.dump(m_en, cell_model_dir / 'model_en.joblib')
    joblib.dump(m_mlp, cell_model_dir / 'model_mlp.joblib')
    joblib.dump(m_knn, cell_model_dir / 'model_knn.joblib')
    joblib.dump(m_ridge, cell_model_dir / 'model_ridge.joblib')
    joblib.dump(m_svr, cell_model_dir / 'model_svr.joblib')
    joblib.dump(meta_model, cell_model_dir / 'meta_model.joblib')

    # ==========================================
    # 6.5 Interpretability & Save Summary
    # ==========================================
    print("  -> Generating feature importance summary...")
    summary_text = f"=====================================\n"
    summary_text += f"    {model_name.upper()} ENSEMBLE SUMMARY\n"
    summary_text += f"=====================================\n\n"

    # Meta-Model Weights
    labels = ["XGB", "LGBM", "RF", "EN", "MLP", "KNN", "Ridge", "SVR"]
    weights = meta_model.coef_
    weight_str = "\n".join([f"{l:10} : {w:.4f}" for l, w in zip(labels, weights)])
    summary_text += f"--- Meta-Model Reliance Weights ---\n{weight_str}\n\n"

    try:
        # Recover Original Feature Names
        gene_cols = pb_tr.columns
        var_mask = pb_selector.named_steps['variance'].get_support()
        kbest_mask = pb_selector.named_steps['kbest'].get_support()
        final_genes = gene_cols[var_mask][kbest_mask]
        pb_feature_names = list(final_genes) + ['sex_binary']
        pc_feature_names = list(pc_tr.columns) + ['sex_binary']

        # --- Weighted Ensemble Features (XGB + RF) ---
        w_xgb = max(0, meta_model.coef_[0])  
        w_rf = max(0, meta_model.coef_[2])
        total_tree_w = w_xgb + w_rf
        
        if total_tree_w > 0:
            weighted_importance = (
                (w_xgb / total_tree_w) * m_xgb.feature_importances_ +
                (w_rf / total_tree_w) * m_rf.feature_importances_
            )
            ensemble_imp_df = pd.Series(weighted_importance, index=pb_feature_names)
            summary_text += "--- Top 15 Overall Genes (Weighted Ensemble XGB+RF) ---\n"
            summary_text += ensemble_imp_df.sort_values(ascending=False).head(15).to_string(float_format="%.5f") + "\n\n"

        # --- ElasticNet (Linear Pseudobulk) ---
        summary_text += "--- Top 15 Features (ElasticNet - Largest Magnitude) ---\n"
        en_imp = pd.Series(np.abs(m_en.coef_), index=pb_feature_names)
        summary_text += en_imp.sort_values(ascending=False).head(15).to_string(float_format="%.5f") + "\n\n"

        # --- Ridge (Genotype PCs) ---
        summary_text += "--- Top 15 Genotype PCs (Ridge - Largest Magnitude) ---\n"
        ridge_imp = pd.Series(np.abs(m_ridge.coef_), index=pc_feature_names)
        summary_text += ridge_imp.sort_values(ascending=False).head(15).to_string(float_format="%.5f") + "\n\n"

    except Exception as e:
        summary_text += f"Could not extract overall feature names due to an error: {e}\n"

    # Save the summary text specifically to this cell type's folder
    summary_path = cell_model_dir / f"{model_name.replace(' ', '_')}_feature_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"  -> Saved feature summary to: {summary_path}")

    # 7. Predict on Test Set
    if test_data_available:
        print("  -> Generating test predictions...")
        p_xgb = m_xgb.predict(pb_te_sel)
        p_lgb = m_lgb.predict(pb_te_sel)
        p_rf  = m_rf.predict(pb_te_sel)
        p_en  = m_en.predict(pb_te_sel)
        p_mlp = m_mlp.predict(gf_te_sel)
        p_knn = m_knn.predict(gf_te_sel)
        p_rid = m_ridge.predict(pc_te_sel)
        p_svr = m_svr.predict(pc_te_sel)
        
        X_meta_te = np.column_stack((p_xgb, p_lgb, p_rf, p_en, p_mlp, p_knn, p_rid, p_svr))
        return meta_model.predict(X_meta_te)
    return None

# ==========================================
# 5. Run Across All Configurations
# ==========================================
CELL_TYPES = ["overall", "CD4 T", "CD8 T", "NK", "B cells", "monocytes"]

# Load Genetic PCs (Constant across all cell types)
print("\nLoading Genotype PCs (Shared baseline)...")
pc_tr_full = pd.concat([load_genotype_pcs("train").loc[y_train.index], load_genotype_pcs("val").loc[y_val.index]]).fillna(0)

if test_data_available:
    pb_test_dummy = load_pseudobulk("test", "overall") # Just to grab test_donors
    test_donors = pb_test_dummy.index
    sex_test = meta.loc[test_donors, ['sex_binary']].fillna(0)
    pc_te_full = load_genotype_pcs("test").loc[test_donors].fillna(0)

test_predictions_dict = {}

for ct in CELL_TYPES:
    # Build Modality Matrices specific to the Cell Type
    pb_train_ct = pd.concat([load_pseudobulk("train", ct).loc[y_train.index], load_pseudobulk("val", ct).loc[y_val.index]]).fillna(0)
    gf_train_ct = pd.concat([load_geneformer("train", ct).loc[y_train.index], load_geneformer("val", ct).loc[y_val.index]]).fillna(0)
    
    if test_data_available:
        pb_test_ct = load_pseudobulk("test", ct).loc[test_donors].fillna(0)
        gf_test_ct = load_geneformer("test", ct).loc[test_donors].fillna(0)
    else:
        pb_test_ct, gf_test_ct = None, None

    # Run encapsulation
    test_preds = train_and_predict_pipeline(
        model_name=ct,
        pb_tr=pb_train_ct, gf_tr=gf_train_ct, pc_tr=pc_tr_full, sex_tr=sex_full, y_tr=y_full,
        pb_te=pb_test_ct, gf_te=gf_test_ct, pc_te=pc_te_full, sex_te=sex_test
    )
    
    if test_data_available:
        test_predictions_dict[ct] = test_preds

# ==========================================
# 6. Save Combined Predictions
# ==========================================
if test_data_available:
    # Rename 'overall' and cell types for clear column headers
    col_names = [f"age_{ct.replace(' ', '_')}" for ct in CELL_TYPES]
    
    submission = pd.DataFrame(test_predictions_dict)
    submission.columns = col_names
    submission.insert(0, 'donor_id', test_donors)
    
    out_path = RESULTS_DIR / "multi_celltype_ensemble_submission.csv"
    submission.to_csv(out_path, index=False)
    
    print("\n=====================================")
    print(f"SUCCESS! Outputted 6 discrete model predictions to:\n {out_path}")
    print(submission.head())