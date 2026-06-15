"""Script to create Kaggle-optimized notebook."""
import json

with open('lstm_text_generator.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update config cell with GPU-optimized hyperparameters
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'BATCH_SIZE' in ''.join(cell['source']):
        src = ''.join(cell['source'])
        src = src.replace('BATCH_SIZE      = 128', 'BATCH_SIZE      = 512   # Larger batch = faster on Kaggle GPU T4')
        src = src.replace('EPOCHS          = 50',  'EPOCHS          = 30    # ~25-40 min on Kaggle GPU')
        cell['source'] = [src]
        print(f'Updated config cell {i}')
        break

# Update bonus epochs for GPU
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'BONUS_EPOCHS = 5' in ''.join(cell['source']):
        src = ''.join(cell['source'])
        src = src.replace('BONUS_EPOCHS = 5    # increase for more thorough comparison',
                          'BONUS_EPOCHS = 10   # 10 epochs per architecture on GPU')
        src = src.replace('BONUS_BATCH  = 256',
                          'BONUS_BATCH  = 512  # Large batch for GPU efficiency')
        cell['source'] = [src]
        print(f'Updated bonus cell {i}')
        break

# GPU check cell to insert after imports
gpu_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        "# ── GPU Check ──────────────────────────────────────────────────────────────\n"
        "import tensorflow as tf\n"
        "gpus = tf.config.list_physical_devices('GPU')\n"
        "print(f'GPUs available: {len(gpus)}')\n"
        "if gpus:\n"
        "    for gpu in gpus:\n"
        "        tf.config.experimental.set_memory_growth(gpu, True)\n"
        "        print(f'  GPU: {gpu.name}')\n"
        "    print('GPU memory growth enabled -- training will be fast!')\n"
        "else:\n"
        "    print('No GPU found! Go to: Settings (right panel) > Accelerator > GPU T4 x2')\n"
    ]
}

# Insert after imports cell (index 2)
nb['cells'].insert(3, gpu_cell)
print('Inserted GPU check cell at position 3')

with open('lstm_text_generator_kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Kaggle notebook saved: lstm_text_generator_kaggle.ipynb')
print('Total cells:', len(nb['cells']))
