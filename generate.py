"""
generate.py
===========
Standalone text generation script for the trained LSTM model.

Usage
-----
    # Generate from a custom seed with default temperature
    python generate.py --seed "to be or not to be"

    # Generate with a specific temperature and length
    python generate.py --seed "shall i compare thee" --temperature 1.2 --length 800

    # Run all sample seeds defined in config.py
    python generate.py --all-seeds

All arguments have defaults from config.py.
"""

import os
import argparse
import re
import numpy as np
import tensorflow as tf

import config
import utils
import model as model_module


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained LSTM model."
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=config.SAMPLE_SEEDS[0],
        help="Seed string to initialise generation"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=config.DEFAULT_TEMPERATURE,
        help="Sampling temperature (default: %(default)s). "
             "Lower → more conservative, higher → more creative."
    )
    parser.add_argument(
        "--length", "-l",
        type=int,
        default=config.GENERATE_LENGTH,
        help="Number of characters to generate (default: %(default)s)"
    )
    parser.add_argument(
        "--architecture", "-a",
        type=str,
        default=config.ARCHITECTURE,
        choices=list(config.ARCHITECTURE_CONFIGS.keys()),
        help="Architecture whose weights to load (default: %(default)s)"
    )
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Generate text for all seeds defined in config.SAMPLE_SEEDS"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Text Generation Core
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_seed(seed: str) -> str:
    """
    Apply the same cleaning steps as the training pipeline to the seed string
    so it is compatible with the character vocabulary.
    """
    seed = seed.lower()
    seed = re.sub(r"[^a-z0-9\n ]", "", seed)
    seed = re.sub(r"[ \t]+", " ", seed)
    return seed.strip()


def generate_text(model: tf.keras.Model,
                  seed: str,
                  char2idx: dict,
                  idx2char: dict,
                  length: int,
                  temperature: float,
                  seq_length: int = config.SEQ_LENGTH
                  ) -> str:
    """
    Generate `length` characters of new text given a seed string.

    Algorithm
    ---------
    1. Normalise the seed and pad/trim to exactly seq_length characters.
    2. Iteratively:
       a. Encode the current window as integer indices.
       b. Feed it through the model to get softmax probabilities.
       c. Sample the next character using temperature scaling.
       d. Append the new character and slide the window one step forward.
    3. Return the generated text.

    Parameters
    ----------
    model       : Trained Keras model
    seed        : Starting text (will be cleaned and trimmed to seq_length)
    char2idx    : Character → index mapping
    idx2char    : Index → character mapping
    length      : Number of new characters to generate
    temperature : Sampling temperature
    seq_length  : Model's expected input sequence length

    Returns
    -------
    str : Seed + newly generated characters
    """
    seed = preprocess_seed(seed)

    # Handle seeds shorter than seq_length: left-pad with space
    if len(seed) < seq_length:
        seed = " " * (seq_length - len(seed)) + seed
    # Handle seeds longer than seq_length: take the last seq_length chars
    elif len(seed) > seq_length:
        seed = seed[-seq_length:]

    # Replace any OOV characters with a space
    seed = "".join(ch if ch in char2idx else " " for ch in seed)

    generated = list(seed)           # mutable buffer
    window    = list(seed)           # sliding context window

    for _ in range(length):
        # Encode the current window
        encoded = np.array([[char2idx[ch] for ch in window]])  # (1, seq_length)

        # Predict next-character probabilities
        predictions = model.predict(encoded, verbose=0)[0]  # (vocab_size,)

        # Sample with temperature
        next_idx  = utils.sample_with_temperature(predictions, temperature)
        next_char = idx2char[next_idx]

        generated.append(next_char)
        window.append(next_char)
        window.pop(0)   # slide the window

    return "".join(generated)


# ─────────────────────────────────────────────────────────────────────────────
# Save Generated Text
# ─────────────────────────────────────────────────────────────────────────────

def save_generated_text(results: list[dict], path: str = config.OUTPUT_TEXT) -> None:
    """
    Save a list of generation results to a formatted text file.

    Each result dict should have keys: seed, temperature, text.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    separator = "=" * 70 + "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write("LSTM TEXT GENERATION — SAMPLE OUTPUTS\n")
        f.write(separator)
        f.write(f"{config.DATASET_CREDIT}\n")
        f.write(separator + "\n")

        for i, result in enumerate(results, start=1):
            f.write(f"SAMPLE {i}\n")
            f.write(f"Seed        : \"{result['seed'][:80]}{'…' if len(result['seed']) > 80 else ''}\"\n")
            f.write(f"Temperature : {result['temperature']}\n")
            f.write(f"Length      : {len(result['text'])} characters\n\n")
            f.write(result["text"])
            f.write("\n\n" + separator + "\n")

    print(f"[INFO] Generated samples saved to '{path}'")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("  LSTM Text Generation — Generate")
    print("=" * 60)
    print(f"  Architecture : {args.architecture}")
    print(f"  Temperature  : {args.temperature}")
    print(f"  Length       : {args.length}")
    print("=" * 60 + "\n")

    # ── Load vocabulary ──────────────────────────────────────────────────────
    char2idx, idx2char, vocab = utils.load_vocab()
    vocab_size = len(vocab)

    # ── Rebuild model and load saved weights ─────────────────────────────────
    model = model_module.build_model(
        vocab_size=vocab_size,
        architecture=args.architecture
    )

    ckpt_path = config.CHECKPOINT_PATH.replace(
        "best_model", f"best_model_{args.architecture}"
    )

    if not os.path.exists(ckpt_path + ".index") and not os.path.exists(ckpt_path):
        print(f"[ERROR] No saved weights found at '{ckpt_path}'.")
        print("        Please run 'python train.py' first.")
        return

    model.load_weights(ckpt_path)
    print(f"[INFO] Loaded weights from '{ckpt_path}'\n")

    # ── Determine seeds ───────────────────────────────────────────────────────
    seeds = config.SAMPLE_SEEDS if args.all_seeds else [args.seed]

    # ── Generate ──────────────────────────────────────────────────────────────
    results = []
    for seed in seeds:
        print(f"[INFO] Generating from seed: \"{seed[:60]}…\"")
        text = generate_text(
            model=model,
            seed=seed,
            char2idx=char2idx,
            idx2char=idx2char,
            length=args.length,
            temperature=args.temperature
        )
        results.append({
            "seed":        seed,
            "temperature": args.temperature,
            "text":        text
        })
        print(f"\n{'─'*60}")
        print(text[:300] + " …")   # Print a preview
        print(f"{'─'*60}\n")

    # ── Save results ──────────────────────────────────────────────────────────
    save_generated_text(results)


if __name__ == "__main__":
    main()
