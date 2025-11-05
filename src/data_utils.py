import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolToSmiles
from tqdm import tqdm
import os
import random
import json
import torch

#-------------------------------------------------------------------------------

def encode_smiles(smiles, token2idx, max_len=128):
    """Encode SMILES into list of token indices with <START> and <END>."""
    tokens = ['<START>'] + list(smiles) + ['<END>']
    ids = [token2idx.get(tok, token2idx['<PAD>']) for tok in tokens]
    if len(ids) < max_len:
        ids += [token2idx['<PAD>']] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

def save_vocab(token2idx, path):
    with open(path, "w") as f:
        json.dump(token2idx, f)
    print(f" Saved vocab at {path}")

def prepare_encoded_dataset(csv_path, vocab_path, out_path, max_len=128):
    """Convert SMILES to token indices and save as torch tensors."""
    df = pd.read_csv(csv_path)
    with open(vocab_path) as f:
        token2idx = json.load(f)
    
    all_encoded = [encode_smiles(smi, token2idx, max_len) for smi in df['canonical']]
    tensor_data = torch.tensor(all_encoded, dtype=torch.long)
    torch.save(tensor_data, out_path)
    print(f" Saved encoded tensor dataset to {out_path}")
def canonicalize_smiles(smiles):
    """Convert SMILES to canonical form using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None
    

#-----------------------------------------------------------------------------------------------

def load_and_clean_dataset(path, sample_size=100000, seed=42):
    """Load SMILES, canonicalize, remove invalid/duplicate molecules."""
    df = pd.read_csv(path)
    col = [c for c in df.columns if 'smile' in c.lower()][0]
    df = df[[col]].rename(columns={col: 'smiles'})
    
    # Random sample if needed
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=seed).reset_index(drop=True)
    
    tqdm.pandas(desc="Canonicalizing SMILES")
    df['canonical'] = df['smiles'].progress_apply(canonicalize_smiles)
    df = df.dropna(subset=['canonical']).drop_duplicates(subset=['canonical']).reset_index(drop=True)
    
    print(f" After cleaning: {len(df)} unique valid molecules.")
    return df

def tokenize_smiles(smiles):
    """Simple character-level tokenization for SMILES."""
    tokens = list(smiles)
    return tokens

def build_vocabulary(smiles_list):
    """Create a token-to-index mapping for model input."""
    vocab = sorted({ch for smi in smiles_list for ch in smi})
    token2idx = {tok: i+2 for i, tok in enumerate(vocab)}  # +2 for PAD/START
    token2idx['<PAD>'] = 0
    token2idx['<START>'] = 1
    token2idx['<END>'] = len(token2idx)
    return token2idx

def split_dataset(df, train_frac=0.9, val_frac=0.05):
    """Split dataframe into train/val/test."""
    train_end = int(len(df) * train_frac)
    val_end = int(len(df) * (train_frac + val_frac))
    return df[:train_end], df[train_end:val_end], df[val_end:]

def save_splits(train, val, test, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(f"{out_dir}/train.csv", index=False)
    val.to_csv(f"{out_dir}/val.csv", index=False)
    test.to_csv(f"{out_dir}/test.csv", index=False)
    print(" Saved splits to:", out_dir)


