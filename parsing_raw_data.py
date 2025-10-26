import os
import gzip
import shutil
from rdkit import Chem
import pandas as pd


INPUT_FOLDER = "C:/Users/dhanu/Desktop/bio-info data"
OUTPUT_FILE = "pubchem_cleaned_molecules.csv"


# STEP 1: Extract .sdf.gz if needed
def decompress_gz(file_path):
    if file_path.endswith(".gz"):
        output_path = file_path[:-3]  # remove .gz
        with gzip.open(file_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"✅ Decompressed: {os.path.basename(output_path)}")
        return output_path
    return file_path

# STEP 2: Parse SDF files
def parse_sdf_file(sdf_path):
    print(f"🔍 Parsing: {os.path.basename(sdf_path)}")
    supplier = Chem.SDMolSupplier(sdf_path, sanitize=False)
    smiles_list = []

    for mol in supplier:
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
        except Exception as e:
            # skip invalid molecules
            continue

    print(f"✅ Parsed {len(smiles_list)} valid molecules from {os.path.basename(sdf_path)}")
    return smiles_list

# STEP 3: Save to CSV
def save_to_csv(smiles_data, output_path):
    df = pd.DataFrame({"smiles": smiles_data})
    df.to_csv(output_path, index=False)
    print(f"💾 Saved {len(df)} molecules to {output_path}")

if __name__ == "__main__":
    all_smiles = []

    for file_name in os.listdir(INPUT_FOLDER):
        if file_name.endswith(".sdf") or file_name.endswith(".sdf.gz"):
            file_path = os.path.join(INPUT_FOLDER, file_name)
            sdf_path = decompress_gz(file_path)
            smiles_list = parse_sdf_file(sdf_path)
            all_smiles.extend(smiles_list)

    if all_smiles:
        save_to_csv(all_smiles, OUTPUT_FILE)
    else:
        print("⚠️ No valid molecules found.")
