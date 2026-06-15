"""
train.py
========
Standalone training script for the LSTM text generation model.

Usage
-----
    python train.py                          # trains the default architecture
    python train.py --architecture deep      # trains the "deep" architecture
    python train.py --architecture wide      # trains the "wide" architecture
    python train.py --epochs 20 --batch 64  # override hyperparameters

All hyperparameter defaults come from config.py.
"""

import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend — safe for scripts without display
import matplotlib.pyplot as plt
import tensorflow as tf

import config
import utils
import model as model_module


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the LSTM character-level text generation model."
    )
    parser.add_argument(
        "--architecture", "-a",
        type=str,
        default=config.ARCHITECTURE,
        choices=list(config.ARCHITECTURE_CONFIGS.keys()),
        help="Model architecture to use (default: %(default)s)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=config.EPOCHS,
        help="Maximum number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=config.BATCH_SIZE,
        help="Mini-batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--seq-length", "-s",
        type=int,
        default=config.SEQ_LENGTH,
        help="Input sequence length (default: %(default)s)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.LEARNING_RATE,
        help="Initial Adam learning rate (default: %(default)s)"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training Callbacks
# ─────────────────────────────────────────────────────────────────────────────

def build_callbacks(checkpoint_path: str) -> list:
    """
    Create a list of Keras training callbacks:
      1. ModelCheckpoint  — saves best weights based on val_loss
      2. EarlyStopping    — halts training when val_loss stops improving
      3. ReduceLROnPlateau — reduces LR when training plateaus
      4. CSVLogger        — appends per-epoch metrics to a CSV file
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    csv_log_path = os.path.join(config.OUTPUTS_DIR, "training_log.csv")
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    callbacks = [
        # Save best model weights only
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        # Stop early if validation loss doesn't improve
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1
        ),
        # Log metrics to CSV for later inspection
        tf.keras.callbacks.CSVLogger(csv_log_path, append=True),
    ]
    return callbacks


# ─────────────────────────────────────────────────────────────────────────────
# Plot Training History
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(history: tf.keras.callbacks.History, architecture: str) -> None:
    """Save a loss and accuracy plot for the training run."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training History — {architecture} architecture", fontsize=14)

    # --- Loss ---
    axes[0].plot(history.history["loss"],     label="Train Loss",      color="#2196F3")
    axes[0].plot(history.history["val_loss"], label="Val Loss",        color="#F44336", linestyle="--")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Sparse Categorical Cross-Entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Accuracy ---
    axes[1].plot(history.history["accuracy"],     label="Train Accuracy", color="#4CAF50")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy",   color="#FF9800", linestyle="--")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUTS_DIR, f"training_history_{architecture}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training history plot saved to '{save_path}'")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  LSTM Text Generation — Training")
    print("=" * 60)
    print(f"  Architecture : {args.architecture}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch}")
    print(f"  Seq length   : {args.seq_length}")
    print(f"  Learning rate: {args.lr}")
    print("=" * 60 + "\n")

    # ── 1. Load and preprocess data ─────────────────────────────────────────
    raw_text   = utils.download_shakespeare()
    clean      = utils.clean_text(raw_text)

    char2idx, idx2char, vocab = utils.build_vocab(clean)
    vocab_size = len(vocab)

    # Persist vocabulary for use during generation
    utils.save_vocab(char2idx, idx2char, vocab)

    # ── 2. Create sequences ─────────────────────────────────────────────────
    X, y = utils.create_sequences(clean, char2idx,
                                  seq_length=args.seq_length,
                                  step=config.STEP_SIZE)

    X_train, X_val, y_train, y_val = utils.train_val_split(X, y)

    # ── 3. Build model ──────────────────────────────────────────────────────
    model = model_module.build_model(
        vocab_size=vocab_size,
        architecture=args.architecture,
        seq_length=args.seq_length,
        learning_rate=args.lr
    )

    # ── 4. Build callbacks ──────────────────────────────────────────────────
    # Checkpoint path includes architecture name to avoid collisions
    ckpt_path = config.CHECKPOINT_PATH.replace(
        "best_model", f"best_model_{args.architecture}"
    )
    callbacks = build_callbacks(ckpt_path)

    # ── 5. Train ────────────────────────────────────────────────────────────
    print("\n[INFO] Starting training …\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # ── 6. Persist history & plot ───────────────────────────────────────────
    history_path = config.HISTORY_PATH.replace(
        "training_history", f"training_history_{args.architecture}"
    )
    np.save(history_path, history.history)
    print(f"[INFO] Training history saved to '{history_path}'")

    plot_history(history, args.architecture)

    # ── 7. Final metrics ────────────────────────────────────────────────────
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n[RESULT] Final Val Loss     : {val_loss:.4f}")
    print(f"[RESULT] Final Val Accuracy : {val_acc:.4f}")
    print(f"\n[INFO] Training complete. Best weights saved to:\n       {ckpt_path}")


if __name__ == "__main__":
    main()
