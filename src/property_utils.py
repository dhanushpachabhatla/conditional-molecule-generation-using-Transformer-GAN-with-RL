from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED
from src import sascorer
import pandas as pd
from tqdm import tqdm
import numpy as np
import json 

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

def compute_properties(smiles):
    """Compute QED, SAS, LogP, TPSA, and MolWt for a given SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        qed = QED.qed(mol)
        sas = sascorer.calculateScore(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        molwt = Descriptors.MolWt(mol)
        return {'QED': qed, 'SAS': sas, 'LogP': logp, 'TPSA': tpsa, 'MolWt': molwt}
    except Exception as e:
        return None

def normalize_properties(df, cols):
    """Min-max normalize selected property columns to 0–1 range."""
    for col in cols:
        if col in df.columns:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
    return df

# def compute_and_save_properties(input_csv, output_csv):
#     """Compute molecular properties for each SMILES and save."""
#     df = pd.read_csv(input_csv)
#     tqdm.pandas(desc="Computing Properties")

#     # Compute properties
#     props = df['canonical'].progress_apply(compute_properties)

#     # Drop None results and reindex properly
#     valid_idx = props[props.notnull()].index
#     props_df = pd.DataFrame(props.loc[valid_idx].tolist())
#     df = df.loc[valid_idx].reset_index(drop=True)
#     df = pd.concat([df, props_df], axis=1)

#     # Ensure columns exist before normalization
#     property_cols = ['QED', 'SAS', 'LogP', 'TPSA', 'MolWt']
#     missing = [c for c in property_cols if c not in df.columns]
#     for m in missing:
#         df[m] = np.nan

#     # Normalize
#     df = normalize_properties(df, property_cols)

#     df.to_csv(output_csv, index=False)
#     print(f" Saved property-augmented dataset to: {output_csv}")
#     print(f" Valid molecules processed: {len(df)} / {len(props)}")
#     return df



def compute_and_save_properties(input_csv, output_csv, stats_path): # <-- 2. ADD 'stats_path' ARGUMENT
    """Compute molecular properties for each SMILES and save."""
    df = pd.read_csv(input_csv)
    tqdm.pandas(desc="Computing Properties")

    # Compute properties
    props = df['canonical'].progress_apply(compute_properties)

    # Drop None results and reindex properly
    valid_idx = props[props.notnull()].index
    props_df = pd.DataFrame(props.loc[valid_idx].tolist())
    df = df.loc[valid_idx].reset_index(drop=True)
    df = pd.concat([df, props_df], axis=1)

    # Ensure columns exist before normalization
    property_cols = ['QED', 'SAS', 'LogP', 'TPSA', 'MolWt']
    missing = [c for c in property_cols if c not in df.columns]
    for m in missing:
        df[m] = np.nan

    # --- 3. NEW BLOCK: Save Un-normalized Stats BEFORE normalizing ---
    prop_stats = {
        'min': df[property_cols].min().to_dict(),
        'max': df[property_cols].max().to_dict()
    }
    with open(stats_path, 'w') as f:
        json.dump(prop_stats, f, indent=4)
    print(f" Saved normalization stats to: {stats_path}")
    # --- END OF NEW BLOCK ---

    # Normalize
    df = normalize_properties(df, property_cols)

    df.to_csv(output_csv, index=False)
    print(f" Saved property-augmented dataset to: {output_csv}")
    print(f" Valid molecules processed: {len(df)} / {len(props)}")
    return df