"""
config.py
=========
Central configuration for the LSTM Text Generation project.
All hyperparameters and file paths are defined here so they can be
changed in one place and propagate throughout the codebase.
"""

import os

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
MODELS_DIR     = os.path.join(BASE_DIR, "models", "checkpoints")
OUTPUTS_DIR    = os.path.join(BASE_DIR, "outputs")

DATA_PATH      = os.path.join(DATA_DIR,    "shakespeare.txt")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "best_model.weights.h5")
HISTORY_PATH   = os.path.join(OUTPUTS_DIR, "training_history.npy")
VOCAB_PATH     = os.path.join(OUTPUTS_DIR, "vocab.pkl")
OUTPUT_TEXT    = os.path.join(OUTPUTS_DIR, "generated_samples.txt")

# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
# Shakespeare's Complete Works from Project Gutenberg
DATASET_URL = (
    "https://www.gutenberg.org/files/100/100-0.txt"
)
DATASET_CREDIT = (
    "Shakespeare's Complete Works — Project Gutenberg\n"
    "URL: https://www.gutenberg.org/ebooks/100\n"
    "License: Public Domain"
)

# ─────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────
SEQ_LENGTH   = 100     # Number of characters in each input sequence
STEP_SIZE    = 3       # Sliding-window step for sequence creation
TRAIN_SPLIT  = 0.80   # Fraction of sequences used for training

# ─────────────────────────────────────────────────────────────
# Model Architectures
# ─────────────────────────────────────────────────────────────
# Available architecture names: "baseline", "deep", "wide"
ARCHITECTURE = "baseline"

ARCHITECTURE_CONFIGS = {
    "baseline": {
        "description": "Single LSTM layer (256 units) — fast training, good quality",
        "lstm_units": [256],
        "dropout_rate": 0.30,
    },
    "deep": {
        "description": "Two stacked LSTM layers (512 -> 256 units) -- richer representation",
        "lstm_units": [512, 256],
        "dropout_rate": 0.30,
    },
    "wide": {
        "description": "Single wide LSTM layer (1024 units) — high capacity",
        "lstm_units": [1024],
        "dropout_rate": 0.35,
    },
}

# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────
BATCH_SIZE     = 128
EPOCHS         = 50
LEARNING_RATE  = 0.001

# Early stopping
EARLY_STOP_PATIENCE  = 5      # Stop if val_loss doesn't improve for N epochs
LR_REDUCE_PATIENCE   = 3      # Halve LR if val_loss doesn't improve for N epochs
LR_REDUCE_FACTOR     = 0.5
MIN_LR               = 1e-6

# ─────────────────────────────────────────────────────────────
# Text Generation
# ─────────────────────────────────────────────────────────────
DEFAULT_SEED_LENGTH  = 100    # Characters used as initial seed
GENERATE_LENGTH      = 500    # Characters to generate after seed
# Temperature controls randomness:
#   < 1.0  → more conservative / repetitive
#   = 1.0  → model's raw distribution
#   > 1.0  → more creative / risky
DEFAULT_TEMPERATURE  = 0.8

# Seeds used in the sample output section
SAMPLE_SEEDS = [
    "to be or not to be that is the question whether tis nobler in the mind to suffer",
    "shall i compare thee to a summers day thou art more lovely and more temperate",
    "all the worlds a stage and all the men and women merely players they have their",
    "friends romans countrymen lend me your ears i come to bury caesar not to praise",
]
