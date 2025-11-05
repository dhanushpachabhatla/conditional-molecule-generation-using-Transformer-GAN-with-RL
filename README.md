# Conditional Molecular Generation using Transformer GAN + RL

## Project Overview
This project aims to generate drug-like molecules conditioned on desired chemical properties using a Transformer-based GAN combined with Reinforcement Learning (RL). The model is trained on SMILES representations of molecules and their calculated properties to produce valid, novel, and optimized molecules.

---

## Current Progress

### 1. Dataset Preparation
- Collected **~8 lakh SMILES strings** from source databases.
- Selected **1 lakh molecules** for initial training.
- Preprocessing includes:
  - **Canonicalization** to standardize molecule representations.
  - **Deduplication** to remove duplicates.
  - **Tokenization** to convert SMILES into discrete tokens.

---

### 2. Property Calculation
- Computed molecular properties using **RDKit** and a custom **SAS scorer**:
  - **QED** (drug-likeness)
  - **SAS** (synthetic accessibility)
  - **LogP** (lipophilicity)
  - **TPSA** (topological polar surface area)
  - **MolWt** (molecular weight)
- Properties are **min-max normalized** and saved in `train_properties.csv`.
- Handled invalid molecules gracefully; warnings generated for problematic structures.

---

### 3. SMILES Encoding
- Built a **vocabulary** mapping each SMILES character to a unique index.
- Converted SMILES sequences into **padded token ID tensors**.
- Saved encoded sequences in **`train_encoded.pt`** for efficient PyTorch training.

---

### 4. Data Loading
- Implemented a **PyTorch Dataset class**:
  - Loads **encoded SMILES sequences** from `.pt` file.
  - Loads **normalized properties** from CSV.
  - Returns **(sequence tensor, property vector)** for each molecule.
- Ready for **batching and GPU training** using PyTorch DataLoader.

---

## Next Steps
1. **Model Implementation**
   - Transformer-based Generator and Discriminator
   - GAN pretraining with Teacher Forcing (MLE)
2. **Adversarial Training**
   - Wasserstein GAN with mini-batch discrimination
3. **RL Fine-tuning**
   - Policy gradient using composite reward (QED, SAS, LogP, etc.)
4. **Conditional Sampling and Evaluation**
   - Generate molecules with specific desired properties
   - Validity, novelty, and property optimization checks
5. **Selection & Ranking**
   - Top candidates for docking or screening

---


