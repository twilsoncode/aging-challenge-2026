import joblib
import pandas as pd
from pathlib import Path

# ==========================================
# 1. Setup Paths
# ==========================================
PROJ_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJ_ROOT / "models" / "saved_final_ensembles"
RESULTS_DIR = PROJ_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# The exact order defined in the training function
BASE_MODEL_NAMES = [
    "XGB (Pseudobulk)", 
    "LGBM (Pseudobulk)", 
    "RF (Pseudobulk)", 
    "ElasticNet (Pseudobulk)", 
    "MLP (Geneformer)", 
    "KNN (Geneformer)", 
    "MLP (scGPT)", 
    "KNN (scGPT)"
]

ENSEMBLES = ["Ensemble_OneK1K", "Ensemble_AIDA", "Ensemble_Both"]

def extract_ensemble_logic():
    all_weights = []
    
    print("Extracting weights from saved ensembles...")
    
    for prefix in ENSEMBLES:
        model_path = MODELS_DIR / prefix / f"{prefix}_meta_ridge.joblib"
        
        if not model_path.exists():
            print(f"  [!] Skipping {prefix}: File not found at {model_path}")
            continue
            
        # Load the Ridge meta-model
        meta_model = joblib.load(model_path)
        
        # Extract coefficients (weights)
        weights = meta_model.coef_
        
        # Create a temporary dataframe for this specific ensemble
        df = pd.DataFrame({
            'Ensemble': prefix,
            'Base_Model': BASE_MODEL_NAMES,
            'Weight': weights
        })
        
        all_weights.append(df)
        print(f"  [✓] Extracted weights for {prefix}")

    if all_weights:
        # Combine and save
        final_df = pd.concat(all_weights, ignore_index=True)
        
        # Sort by weight magnitude for easier interpretation
        final_df['Abs_Weight'] = final_df['Weight'].abs()
        final_df = final_df.sort_values(by=['Ensemble', 'Abs_Weight'], ascending=[True, False]).drop(columns=['Abs_Weight'])
        
        output_path = RESULTS_DIR / "ensemble_reliance_weights.csv"
        final_df.to_csv(output_path, index=False)
        
        print(f"\nSuccess! Master weights report saved to: {output_path}")
        print("\nTop 3 Contributors per Ensemble:")
        for ens in ENSEMBLES:
            top = final_df[final_df['Ensemble'] == ens].head(3)
            if not top.empty:
                print(f"\n--- {ens} ---")
                print(top[['Base_Model', 'Weight']].to_string(index=False))
    else:
        print("\nNo models found. Please check your 'models/saved_final_ensembles' directory.")

if __name__ == "__main__":
    extract_ensemble_logic()