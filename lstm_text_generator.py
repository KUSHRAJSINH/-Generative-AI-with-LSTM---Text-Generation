import os
import re
import pickle
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import layers, callbacks as keras_callbacks

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"TensorFlow : {tf.__version__}")
print(f"NumPy      : {np.__version__}")
print(f"GPU        : {bool(tf.config.list_physical_devices('GPU'))}\n")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models", "checkpoints")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

DATA_PATH   = os.path.join(DATA_DIR, "shakespeare.txt")
VOCAB_PATH  = os.path.join(OUTPUTS_DIR, "vocab.pkl")
OUTPUT_TEXT = os.path.join(OUTPUTS_DIR, "generated_samples.txt")

DATASET_URL = "https://www.gutenberg.org/files/100/100-0.txt"

SEQ_LENGTH  = 100
STEP_SIZE   = 3
TRAIN_SPLIT = 0.80

BATCH_SIZE     = 128
EPOCHS         = 50
LEARNING_RATE  = 0.001
EARLY_PATIENCE = 5
LR_PATIENCE    = 3
LR_FACTOR      = 0.5
MIN_LR         = 1e-6

GENERATE_LENGTH = 500
TEMPERATURES    = [0.2, 0.8, 1.2]

SAMPLE_SEEDS = [
    "to be or not to be that is the question whether tis nobler in the mind to suffer",
    "shall i compare thee to a summers day thou art more lovely and more temperate",
    "all the worlds a stage and all the men and women merely players they have their",
    "friends romans countrymen lend me your ears i come to bury caesar not to praise",
]

def download_shakespeare(url=DATASET_URL, save_path=DATA_PATH):
    if os.path.exists(save_path):
        print(f"Loading dataset from '{save_path}'...")
        with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    print("Downloading dataset...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    raw = response.text
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"Downloaded {len(raw):,} chars")
    return raw

raw_text = download_shakespeare()
print(f"Total characters: {len(raw_text):,}")

def clean_text(text):
    s = text.find("*** START OF")
    e = text.find("*** END OF")
    if s != -1:
        text = text[text.find("\n", s) + 1:]
    if e != -1:
        text = text[:e]

    text = text.lower()
    text = re.sub(r"[^a-z0-9\n ]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

clean = clean_text(raw_text)
print(f"Cleaned chars: {len(clean):,}")

def build_vocab(text):
    vocab = sorted(set(text))
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    return char2idx, idx2char, vocab

char2idx, idx2char, vocab = build_vocab(clean)
VOCAB_SIZE = len(vocab)
print(f"Vocab size: {VOCAB_SIZE}")

with open(VOCAB_PATH, "wb") as f:
    pickle.dump({"char2idx": char2idx, "idx2char": idx2char, "vocab": vocab}, f)

def create_sequences(text, char2idx, seq_length=SEQ_LENGTH, step=STEP_SIZE):
    encoded = np.array([char2idx[ch] for ch in text], dtype=np.int32)
    starts = list(range(0, len(encoded) - seq_length, step))

    X = np.stack([encoded[i : i + seq_length] for i in tqdm(starts, desc="Sequences")])
    y = np.array([encoded[i + seq_length] for i in starts], dtype=np.int32)
    return X, y

X, y = create_sequences(clean, char2idx)

def train_val_split(X, y, train_ratio=TRAIN_SPLIT, seed=SEED):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    split = int(len(X) * train_ratio)
    tr, va = indices[:split], indices[split:]
    return X[tr], X[va], y[tr], y[va]

X_train, X_val, y_train, y_val = train_val_split(X, y)
print(f"Train/Val: {len(X_train)} / {len(X_val)}")

ARCHITECTURE_CONFIGS = {
    "baseline": {"lstm_units": [256], "dropout_rate": 0.30},
    "deep": {"lstm_units": [512, 256], "dropout_rate": 0.30},
    "wide": {"lstm_units": [1024], "dropout_rate": 0.35},
}

def build_model(vocab_size, architecture="baseline", seq_length=SEQ_LENGTH, lr=LEARNING_RATE):
    cfg = ARCHITECTURE_CONFIGS[architecture]
    units = cfg["lstm_units"]
    drop = cfg["dropout_rate"]

    embed_dim = min(128, int(vocab_size ** 0.5) * 4)
    inputs = tf.keras.Input(shape=(seq_length,), name="char_indices")
    x = layers.Embedding(vocab_size, embed_dim, name="char_embedding")(inputs)

    for i, n_units in enumerate(units):
        is_last = (i == len(units) - 1)
        x = layers.LSTM(n_units, return_sequences=not is_last, name=f"lstm_{i+1}")(x)
        x = layers.Dropout(drop, name=f"dropout_{i+1}")(x)

    outputs = layers.Dense(vocab_size, activation="softmax", name="char_predictions")(x)
    model = tf.keras.Model(inputs, outputs, name=f"lstm_{architecture}")
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=["accuracy"]
    )
    return model

model = build_model(VOCAB_SIZE, architecture="baseline")
model.summary()

CKPT_PATH = os.path.join(MODELS_DIR, "best_model_baseline.weights.h5")
CSV_LOG = os.path.join(OUTPUTS_DIR, "training_log.csv")

training_callbacks = [
    keras_callbacks.ModelCheckpoint(CKPT_PATH, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1),
    keras_callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_PATIENCE, restore_best_weights=True, verbose=1),
    keras_callbacks.ReduceLROnPlateau(monitor="val_loss", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR, verbose=1),
    keras_callbacks.CSVLogger(CSV_LOG, append=False),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=training_callbacks,
    verbose=1
)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

def plot_training_history(hist, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(hist["loss"], label="Train")
    axes[0].plot(hist["val_loss"], label="Val", ls="--")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(hist["accuracy"], label="Train")
    axes[1].plot(hist["val_accuracy"], label="Val", ls="--")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

plot_training_history(history.history, save_path=os.path.join(OUTPUTS_DIR, "training_history_baseline.png"))

def sample_with_temperature(preds, temperature=1.0):
    preds = preds.astype(np.float64)
    log_p = np.log(preds + 1e-8) / temperature
    log_p -= log_p.max()
    exp_p = np.exp(log_p)
    p = exp_p / exp_p.sum()
    return int(np.random.choice(len(p), p=p))

def generate_text(model, seed, char2idx, idx2char, length=GENERATE_LENGTH, temperature=0.8, seq_length=SEQ_LENGTH):
    seed = seed.lower()
    seed = re.sub(r"[^a-z0-9\n ]", "", seed).strip()

    if len(seed) < seq_length:
        seed = " " * (seq_length - len(seed)) + seed
    elif len(seed) > seq_length:
        seed = seed[-seq_length:]

    seed = "".join(ch if ch in char2idx else " " for ch in seed)
    window = list(seed)
    generated = list(seed)

    for _ in range(length):
        encoded = np.array([[char2idx[c] for c in window]])
        preds = model.predict(encoded, verbose=0)[0]
        next_char = idx2char[sample_with_temperature(preds, temperature)]
        generated.append(next_char)
        window.append(next_char)
        window.pop(0)

    return "".join(generated)

all_results = []
for seed_idx, seed in enumerate(SAMPLE_SEEDS, 1):
    print(f"\nGenerating for seed {seed_idx}...")
    for temp in TEMPERATURES:
        generated = generate_text(model, seed, char2idx, idx2char, length=GENERATE_LENGTH, temperature=temp)
        all_results.append({"seed": seed, "temperature": temp, "text": generated})

with open(OUTPUT_TEXT, "w", encoding="utf-8") as f:
    for i, r in enumerate(all_results, 1):
        f.write(f"SAMPLE {i}\nSeed: {r['seed']}\nTemp: {r['temperature']}\n\n{r['text']}\n\n{'='*50}\n")

print(f"Saved text to {OUTPUT_TEXT}")

BONUS_EPOCHS = 5
BONUS_BATCH = 256
COMPARISON_RESULTS = {}

for arch_name, arch_cfg in ARCHITECTURE_CONFIGS.items():
    print(f"Training '{arch_name}'...")
    m = build_model(VOCAB_SIZE, architecture=arch_name)
    ckpt = os.path.join(MODELS_DIR, f"best_model_{arch_name}.weights.h5")
    cb = [
        keras_callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=0),
        keras_callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0),
        keras_callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=0),
    ]
    h = m.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=BONUS_EPOCHS, batch_size=BONUS_BATCH, callbacks=cb, verbose=0)
    vl, va = m.evaluate(X_val, y_val, verbose=0)
    COMPARISON_RESULTS[arch_name] = {"history": h.history, "val_loss": vl, "val_accuracy": va, "perplexity": np.exp(vl), "params": m.count_params(), "model": m}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for arch, res in COMPARISON_RESULTS.items():
    axes[0].plot(res["history"]["val_loss"], label=arch)
axes[0].set_title("Val Loss")
axes[0].legend()

archs = list(COMPARISON_RESULTS.keys())
accs = [COMPARISON_RESULTS[a]["val_accuracy"] for a in archs]
axes[1].bar(archs, accs)
axes[1].set_title("Val Accuracy")

perps = [COMPARISON_RESULTS[a]["perplexity"] for a in archs]
axes[2].bar(archs, perps)
axes[2].set_title("Perplexity")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "architecture_comparison.png"), dpi=150)
plt.close()

bonus_output_path = os.path.join(OUTPUTS_DIR, "bonus_architecture_comparison.txt")
with open(bonus_output_path, "w", encoding="utf-8") as f:
    for arch, res in COMPARISON_RESULTS.items():
        gen = generate_text(res["model"], SAMPLE_SEEDS[0], char2idx, idx2char, length=300, temperature=0.8)
        f.write(f"--- {arch.upper()} ---\n{gen}\n\n")

print("Done")
