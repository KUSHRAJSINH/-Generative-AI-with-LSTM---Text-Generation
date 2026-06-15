"""
utils.py
========
Data preprocessing and vocabulary utilities for LSTM text generation.

Functions
---------
download_shakespeare()    → downloads corpus from Project Gutenberg
clean_text(text)          → normalises raw text
build_vocab(text)         → builds character ↔ integer mappings
create_sequences(text)    → sliding-window (X, y) pairs
save_vocab / load_vocab   → persistence helpers
"""

import os
import re
import pickle
import requests
import numpy as np
from tqdm import tqdm

import config


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset Acquisition
# ─────────────────────────────────────────────────────────────────────────────

def download_shakespeare(url: str = config.DATASET_URL,
                         save_path: str = config.DATA_PATH) -> str:
    """
    Download Shakespeare's complete works from Project Gutenberg and save to disk.

    Parameters
    ----------
    url       : Source URL (default from config.DATASET_URL)
    save_path : Where to save the .txt file (default from config.DATA_PATH)

    Returns
    -------
    str : The raw downloaded text
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"[INFO] Dataset already exists at '{save_path}'. Loading from disk.")
        with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    print(f"[INFO] Downloading dataset from:\n       {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    raw_text = response.text

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    print(f"[INFO] Saved {len(raw_text):,} characters to '{save_path}'")
    return raw_text


# ─────────────────────────────────────────────────────────────────────────────
# 2. Text Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise raw corpus text for character-level modelling.

    Steps
    -----
    1. Strip Project Gutenberg header/footer boilerplate.
    2. Convert to lowercase.
    3. Remove punctuation (keep only letters, digits, whitespace).
    4. Collapse multiple whitespace characters into a single space.

    Parameters
    ----------
    text : Raw text string

    Returns
    -------
    str : Cleaned text
    """
    # --- Strip Gutenberg boilerplate ---
    # Header ends at the first occurrence of "*** START OF"
    start_marker = "*** START OF"
    end_marker   = "*** END OF"

    start_idx = text.find(start_marker)
    end_idx   = text.find(end_marker)

    if start_idx != -1:
        # Move past the header line
        text = text[text.find("\n", start_idx) + 1 :]
    if end_idx != -1:
        text = text[:end_idx]

    # --- Lowercase ---
    text = text.lower()

    # --- Remove punctuation (keep letters, digits, newlines, spaces) ---
    text = re.sub(r"[^a-z0-9\n ]", "", text)

    # --- Collapse whitespace ---
    text = re.sub(r"[ \t]+", " ", text)   # multiple spaces → single space
    text = re.sub(r"\n+",    "\n", text)  # multiple newlines → single newline
    text = text.strip()

    print(f"[INFO] Cleaned text length: {len(text):,} characters")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 3. Vocabulary Building
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab(text: str) -> tuple[dict, dict, list]:
    """
    Build character-level vocabulary from the cleaned corpus.

    Returns
    -------
    char2idx : dict mapping character → integer index
    idx2char : dict mapping integer index → character
    vocab    : sorted list of unique characters
    """
    vocab    = sorted(set(text))
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = {i: ch for i, ch in enumerate(vocab)}

    print(f"[INFO] Vocabulary size: {len(vocab)} unique characters")
    print(f"[INFO] Characters: {''.join(vocab)!r}")
    return char2idx, idx2char, vocab


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sequence Creation
# ─────────────────────────────────────────────────────────────────────────────

def create_sequences(text: str,
                     char2idx: dict,
                     seq_length: int = config.SEQ_LENGTH,
                     step: int = config.STEP_SIZE
                     ) -> tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping input/output character-index sequences via a sliding window.

    Each sample:
      Input  → integer indices for characters [i : i + seq_length]
      Output → integer index for character at position [i + seq_length]

    Parameters
    ----------
    text       : Cleaned corpus string
    char2idx   : Character-to-index mapping
    seq_length : Length of each input sequence
    step       : Stride of the sliding window

    Returns
    -------
    X : np.ndarray of shape (n_samples, seq_length) — integer indices
    y : np.ndarray of shape (n_samples,)             — integer target index
    """
    print(f"[INFO] Creating sequences (seq_len={seq_length}, step={step}) …")

    # Pre-encode the entire text once to avoid per-character dict look-ups
    encoded = np.array([char2idx[ch] for ch in text], dtype=np.int32)

    # Build list of start positions
    starts = range(0, len(encoded) - seq_length, step)

    X = np.stack([encoded[i : i + seq_length] for i in tqdm(starts,
                  desc="Building sequences", unit="seq")])
    y = np.array([encoded[i + seq_length]     for i in starts], dtype=np.int32)

    print(f"[INFO] Created {len(X):,} sequences")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 5. Train / Validation Split
# ─────────────────────────────────────────────────────────────────────────────

def train_val_split(X: np.ndarray,
                    y: np.ndarray,
                    train_ratio: float = config.TRAIN_SPLIT,
                    shuffle: bool = True,
                    seed: int = 42
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequences into training and validation sets.

    Parameters
    ----------
    X           : Input sequences
    y           : Target indices
    train_ratio : Fraction to use for training (rest goes to validation)
    shuffle     : Whether to shuffle before splitting
    seed        : Random seed for reproducibility

    Returns
    -------
    X_train, X_val, y_train, y_val
    """
    n = len(X)
    indices = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    split = int(n * train_ratio)
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"[INFO] Train samples : {len(X_train):,}")
    print(f"[INFO] Val   samples : {len(X_val):,}")
    return X_train, X_val, y_train, y_val


# ─────────────────────────────────────────────────────────────────────────────
# 6. Vocabulary Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_vocab(char2idx: dict, idx2char: dict, vocab: list,
               path: str = config.VOCAB_PATH) -> None:
    """Pickle vocabulary mappings to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"char2idx": char2idx,
                     "idx2char": idx2char,
                     "vocab":    vocab}, f)
    print(f"[INFO] Vocabulary saved to '{path}'")


def load_vocab(path: str = config.VOCAB_PATH) -> tuple[dict, dict, list]:
    """Load vocabulary mappings from disk."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[INFO] Vocabulary loaded from '{path}'")
    return data["char2idx"], data["idx2char"], data["vocab"]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Text Generation Helper
# ─────────────────────────────────────────────────────────────────────────────

def sample_with_temperature(predictions: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample a character index from the model's output probability distribution
    with temperature scaling.

    Temperature
    -----------
    < 1.0 → sharpen distribution (more deterministic / conservative)
    = 1.0 → use raw softmax probabilities
    > 1.0 → flatten distribution (more diverse / random)

    Parameters
    ----------
    predictions : 1-D array of logits or probabilities (length = vocab_size)
    temperature : Scaling factor

    Returns
    -------
    int : Sampled character index
    """
    predictions = predictions.astype(np.float64)

    # Re-scale via log → temperature → softmax
    log_preds = np.log(predictions + 1e-8) / temperature
    # Subtract max for numerical stability before exp
    log_preds -= log_preds.max()
    preds_exp  = np.exp(log_preds)
    preds_norm = preds_exp / preds_exp.sum()

    return int(np.random.choice(len(preds_norm), p=preds_norm))
