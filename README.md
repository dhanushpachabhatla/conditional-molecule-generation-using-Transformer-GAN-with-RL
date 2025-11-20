# Conditional Molecular Generation with GANs and Reinforcement Learning

This project implements a generative model for creating novel, valid chemical molecules (represented as SMILES strings). The model is a Transformer-based Generative Adversarial Network (GAN) trained with Reinforcement Learning (RL) to generate molecules that match specific, desired chemical properties (such as QED, LogP, and SAS).

The model can be operated in two modes:
1.  **Conditional Generation:** The user provides a vector of target properties, and the model generates a new molecule that attempts to match them.
2.  **Unconditional Generation:** The user provides no properties, and the model generates a new, valid, and random molecule from its learned chemical space.



## 🧪 Methodology

The model is trained in a three-phase pipeline:
1.  **Generator Pre-training (Supervised):** A "decoder-only" Transformer (like GPT) is pre-trained on a large dataset of known molecules. It learns the "grammar" of SMILES strings and how to generate valid structures. This step uses **conditional dropping** to train the model for both conditional and unconditional tasks simultaneously.
2.  **Discriminator Pre-training (Supervised):** A separate "encoder" Transformer (like BERT) is pre-trained as a critic. It is fed a mix of "real" molecules from the dataset and "fake" molecules from the pre-trained generator. Its only job is to get exceptionally good at distinguishing real from fake.
3.  **RL Fine-Tuning (GAN Training):** This is the core GAN-RL loop. The pre-trained Generator (the "Policy") and the pre-trained Discriminator (the "Critic") are combined.
    * The Generator generates a batch of molecules.
    * A **Total Reward** is calculated based on two scores:
        1.  **Critic Reward (`reward_D`):** How "real" does the critic think this molecule is? (from the Discriminator)
        2.  **Property Reward (`reward_P`):** How well does this molecule match the user's target properties? (from RDKit)
    * The total reward is `Total Reward = (W_DISC * reward_D) + (W_PROP * reward_P)`.
    * The Generator's weights are updated using a policy gradient algorithm (REINFORCE) to maximize this total reward.

## 📦 Project Structure

```
project_root/
├── data/
│   ├── raw/
│   │   └── smiles_raw.csv         #  initial raw SMILES dataset (downloaded from NCBI FTP)
│   └── processed_5l/
|       ├── train.csv   (from notebook 1)
|       ├── test.csv   (from notebook 1)
|       ├── val.csv   (from notebook 1)
│       ├── train_properties.csv   # Normalized training set
│       ├── val_properties.csv     # Normalized validation set
│       ├── test_properties.csv    # Normalized test set 
│       ├── train_encoded.pt       # Tokenized training sequences
│       ├── val_encoded.pt         # Tokenized validation sequences
│       ├── prop_stats.json        #  Un-normalized min/max stats
│       └── vocab_5l.json             # Token-to-ID mapping
│
├── notebooks/
│   ├── 01_preprocess.ipynb
│   ├── 02_pretrain_generator.ipynb
│   ├── 03_validate_generator.ipynb
│   ├── 04_pretrain_discriminator.ipynb
│   ├── 05_validate_discriminator.ipynb
│   ├── 06_GAN_finetuning_RL.ipynb        # The final RL training loop
│   └── 07_sampling_evaluation.ipynb  # To test the final RL-tuned model
│
├── results/
│   └── models_5l/
│       ├── u&c_generator_epoch_50.pt  # Pre-trained Generator (from notebook 2)
│       ├── discriminator_epoch_1.pt   # Pre-trained Discriminator (from notebook 4)
│       └── generator_RL_step_5000.pt  # Final, fine-tuned Generator (from notebook 6)
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py          # For tokenizing and creating datasets
│   ├── property_utils.py      # For calculating properties (QED, etc.)
│   └── sascorer.py            # SA Score calculation utility
│
└── requirements.txt
```

## 🚀 Project Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dhanushpachabhatla/conditional-molecule-generation-using-Transformer-GAN-with-RL.git
    cd conditional-molecule-generation-using-Transformer-GAN-with-RL
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    .\venv\Scripts\activate   # On Windows
    ```

3.  **Install the required packages:**
    A `requirements.txt` file is provided.
    ```bash
    pip install -r requirements.txt
    ```
    Or, install the key packages manually:
    ```
    pip install torch pandas numpy tqdm
    pip install rdkit-pypi
    pip install sascorer
    ```

## 🏃‍♂️ Order of Execution

This project must be run in sequence.

### 1. `01_preprocess.ipynb`
* **Goal:** To process your raw SMILES data.
* **Action:** Loads `data/raw/smiles_raw.csv`, samples 300k+ molecules, cleans them, and computes properties.
* **Output:** Creates all the files in `data/processed_5l/`, including the critical `prop_stats.json`.

### 2. `02_pretrain_generator.ipynb`
* **Goal:** To train the Generator to create valid, diverse molecules.
* **Action:** Loads the processed data and trains the `Generator` model using conditional dropping (`p_uncond = 0.1`).
* **Output:** `results/models_5l/u&c_generator_epoch_50.pt`.

### 3. `03_validate_generator.ipynb`
* **Goal:** To verify the Generator is "smart" before continuing.
* **Action:** Loads the generator from Step 2. Generates 1,000+ molecules and uses RDKit to check their quality.
* **Expected Result:** High validity (>70%) for both conditional and unconditional generation.

### 4. `04_pretrain_discriminator.ipynb`
* **Goal:** To train the Discriminator to be a "master critic."
* **Action:** Loads "real" data from `train_encoded.pt` and generates "fake" data on-the-fly using the generator from Step 2. Trains the `Discriminator` to tell them apart.
* **Output:** `results/models_5l/discriminator_epoch_1.pt`.

### 5. `05_validate_discriminator.ipynb`
* **Goal:** To verify the Discriminator is "smart."
* **Action:** Loads the critic from Step 4. Tests its accuracy against unseen real and fake molecules.
* **Expected Result:** Very high total accuracy (>95-98%).

### 6. `06_gan_training.ipynb`
* **Goal:** To fine-tune the Generator to match specific properties.
* **Action:** This is the final RL loop. It loads the Generator (from Step 2) and the Discriminator (from Step 4). It freezes the Discriminator and uses it as a reward function to train the Generator.
* **Output:** A final, fine-tuned model: `results/models_5l/generator_RL_step_5000.pt`.

### 7. `07_validate_final_model.ipynb`
* **Goal:** To test the final, fine-tuned model.
* **Action:** Load the RL-tuned generator from Step 6. Give it specific property targets and see how well the generated molecules match those properties.

## 📊 Results (As of 2025-11-18)

This section summarizes the results from the project's development.

### Generator Pre-training
* **Dataset:** 300,000 molecules
* **Epochs:** 50
* **Final Loss:** ~0.64
* **Result:** A stable, "creative" generator.

### Generator Validation (Before RL)
| Generation Mode | Validity | Uniqueness | Novelty |
| :--- | :---: | :---: | :---: |
| **Unconditional** | 81.35% | 99.76% | 92.06% |
| **Conditional (In-Distribution)** | 75.78% | 100.00% | 99.74% |

### Discriminator Pre-training
* **Epochs:** 1
* **Final Loss:** ~0.028
* **Validation Accuracy (on unseen data):** **98.68%**
* **Conclusion:** The discriminator is a "master critic" and is ready for RL.

### RL Fine-Tuning
* **Initial Run (W: 0.5/0.5):** The model successfully learned to fool the critic (`reward_D` increased to `0.439`) but **failed** to learn property matching (`reward_P` worsened to `-0.089`). This is a classic case of **reward hacking**.
* **Current Run (W: 0.2/0.8):** The training is now correctly balanced and changed the logic

# results for 1984 gen conditional samples
```bash
* === QUALITY METRICS ===
Validity:   86.49%
Uniqueness: 99.50%
Novelty:    95.61%
```

# property distribution of 1984 mols generated
<img width="1149" height="718" alt="image" src="https://github.com/user-attachments/assets/83aa4460-b8be-4844-9185-006cfa2c1584" />

# results for 50 gen conditional samples
```bash
Mean MSE: 0.0251
Median MSE: 0.0024
Mean Cosine Similarity: 0.9353
Validity: 86.00%
```
