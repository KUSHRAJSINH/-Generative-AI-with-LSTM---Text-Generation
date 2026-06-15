"""Add a zip/download cell to the Kaggle notebook."""
import json

with open('lstm_text_generator_kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

zip_src = """\
# ── DOWNLOAD HELPER: Zip all results ──────────────────────────────────────
# After this cell runs, find lstm_results.zip in the Output tab (right sidebar)
import shutil, os

# List all saved files with sizes
print("All saved files:")
for root, dirs, files in os.walk("/kaggle/working"):
    for fname in sorted(files):
        if not fname.endswith(".zip"):
            path = os.path.join(root, fname)
            size_kb = os.path.getsize(path) / 1024
            short = path.replace("/kaggle/working/", "")
            print(f"  {short:<55} {size_kb:>8.1f} KB")

# Zip everything into one file
shutil.make_archive("/kaggle/working/lstm_results", "zip", "/kaggle/working")
zip_mb = os.path.getsize("/kaggle/working/lstm_results.zip") / (1024 * 1024)
print(f"\\nCreated lstm_results.zip  ({zip_mb:.1f} MB)")
print("Download from: Output tab (right sidebar) -> lstm_results.zip")
print("\\nContains:")
print("  models/checkpoints/*.weights.h5   <- trained model weights")
print("  outputs/generated_samples.txt     <- generated Shakespeare text")
print("  outputs/training_history_*.png    <- training curves")
print("  outputs/architecture_comparison.png <- bonus comparison plot")
"""

zip_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [zip_src]
}

nb['cells'].append(zip_cell)

with open('lstm_text_generator_kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Added zip/download cell.')
print('Total cells:', len(nb['cells']))
