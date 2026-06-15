"""
model.py
========
LSTM model architecture definitions for character-level text generation.

Three configurable architectures are provided:
  • baseline : Single LSTM(256) layer — quick to train, solid quality
  • deep      : Stacked LSTM(512 → 256) — richer sequential representations
  • wide      : Single LSTM(1024) layer — high-capacity single layer

All models share the same interface:
  build_model(vocab_size, architecture_name) → tf.keras.Model
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import config


# ─────────────────────────────────────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(vocab_size: int,
                architecture: str = config.ARCHITECTURE,
                seq_length: int   = config.SEQ_LENGTH,
                learning_rate: float = config.LEARNING_RATE
                ) -> tf.keras.Model:
    """
    Build and compile an LSTM text generation model.

    Architecture
    ------------
    Input (seq_length,)
      → Embedding(vocab_size, embed_dim)
      → LSTM layer(s) with dropout
      → Dense(vocab_size, activation='softmax')

    Parameters
    ----------
    vocab_size    : Number of unique characters in the vocabulary
    architecture  : One of "baseline", "deep", or "wide"
    seq_length    : Length of input character sequences
    learning_rate : Adam optimiser learning rate

    Returns
    -------
    tf.keras.Model : Compiled Keras model ready for training
    """
    arch_cfg = config.ARCHITECTURE_CONFIGS[architecture]
    lstm_units   = arch_cfg["lstm_units"]
    dropout_rate = arch_cfg["dropout_rate"]
    description  = arch_cfg["description"]

    print(f"\n[INFO] Building '{architecture}' model -- {description}")
    print(f"       LSTM units  : {lstm_units}")
    print(f"       Dropout     : {dropout_rate}")
    print(f"       Vocab size  : {vocab_size}\n")

    # ── Embedding dimension: square-root heuristic, capped at 128 ──────────
    embed_dim = min(128, int(vocab_size ** 0.5) * 4)

    # ── Input ───────────────────────────────────────────────────────────────
    inputs = tf.keras.Input(shape=(seq_length,), name="char_indices")

    # ── Embedding Layer ─────────────────────────────────────────────────────
    # Converts integer character indices to dense vectors
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        name="char_embedding"
    )(inputs)

    # ── LSTM Layers ─────────────────────────────────────────────────────────
    for i, units in enumerate(lstm_units):
        is_last_lstm = (i == len(lstm_units) - 1)

        # return_sequences=True for all intermediate LSTM layers
        # return_sequences=False for the final LSTM layer
        x = layers.LSTM(
            units=units,
            return_sequences=not is_last_lstm,
            name=f"lstm_{i+1}"
        )(x)

        # Dropout after each LSTM to reduce overfitting
        x = layers.Dropout(rate=dropout_rate, name=f"dropout_{i+1}")(x)

    # ── Output Layer ────────────────────────────────────────────────────────
    # Dense + softmax predicts probability over each character
    outputs = layers.Dense(
        units=vocab_size,
        activation="softmax",
        name="char_predictions"
    )(x)

    # ── Assemble Model ──────────────────────────────────────────────────────
    model = models.Model(inputs=inputs, outputs=outputs,
                         name=f"lstm_textgen_{architecture}")

    # ── Compile ─────────────────────────────────────────────────────────────
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"]
    )

    model.summary()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: Build all three architectures (used in bonus comparison)
# ─────────────────────────────────────────────────────────────────────────────

def build_all_architectures(vocab_size: int,
                             seq_length: int = config.SEQ_LENGTH,
                             learning_rate: float = config.LEARNING_RATE
                             ) -> dict:
    """
    Build all three architecture variants and return them in a dictionary.

    Returns
    -------
    dict : {"baseline": model, "deep": model, "wide": model}
    """
    return {
        name: build_model(vocab_size, name, seq_length, learning_rate)
        for name in config.ARCHITECTURE_CONFIGS
    }
