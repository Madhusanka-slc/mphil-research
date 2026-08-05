"""
Update all training notebooks to include EPOCH and other configuration parameters.
Run this script to add configuration cells to all notebooks.
"""

import json
from pathlib import Path

def create_config_cell(epochs, batch_size, lr, img_size=256):
    """Create a configuration cell for a notebook."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ===== TRAINING CONFIGURATION (Change these values) =====\n",
            f"EPOCHS = {epochs}              # Number of training epochs\n",
            f"BATCH_SIZE = {batch_size}           # Batch size\n",
            f"LEARNING_RATE = {lr}    # Learning rate\n",
            f"IMAGE_SIZE = {img_size}        # Input image size\n",
            "# =====================================================\n",
            "\n",
            "print(f'Configuration:')\n",
            "print(f'  EPOCHS: {EPOCHS}')\n",
            "print(f'  BATCH_SIZE: {BATCH_SIZE}')\n",
            "print(f'  LEARNING_RATE: {LEARNING_RATE}')\n",
            "print(f'  IMAGE_SIZE: {IMAGE_SIZE}x{IMAGE_SIZE}')\n"
        ]
    }

# Notebook configurations
notebooks = {
    "experiments/1_simple_unet_learning/simple_unet_colab.ipynb": {
        "epochs": 20,
        "batch": 4,
        "lr": "1e-3",
        "img_size": 256
    },
    "experiments/2_cgan/colab_train.ipynb": {
        "epochs": 50,
        "batch": 4,
        "lr": "2e-4",
        "img_size": 256
    },
    "experiments/3_yolov8/colab_train.ipynb": {
        "epochs": 50,
        "batch": 8,
        "lr": "1e-3",
        "img_size": 640
    },
    "experiments/4_fpn_unet_swin/02_train_colab.ipynb": {
        "epochs": 50,
        "batch": 8,
        "lr": "1e-3",
        "img_size": 384
    },
    "experiments/5_sam_vmnet/02_sam_vmnet_finetune.ipynb": {
        "epochs": 50,
        "batch": 4,
        "lr": "1e-4",
        "img_size": 512
    }
}

print("\n" + "="*60)
print("UPDATE TRAINING NOTEBOOKS WITH CONFIGURATION")
print("="*60)

for notebook_path, config in notebooks.items():
    full_path = Path(notebook_path)

    if not full_path.exists():
        print(f"⚠ NOT FOUND: {notebook_path}")
        continue

    print(f"\nProcessing: {notebook_path}")

    try:
        # Load notebook
        with open(full_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        # Create config cell
        config_cell = create_config_cell(
            config["epochs"],
            config["batch"],
            config["lr"],
            config["img_size"]
        )

        # Find insertion point (after imports and base setup)
        insert_index = 1
        for i, cell in enumerate(nb['cells']):
            if cell.get('cell_type') == 'code':
                source_text = ''.join(cell.get('source', []))
                # Look for BASE_URL or similar setup
                if 'BASE_URL' in source_text or 'import' in source_text:
                    insert_index = i + 1
                else:
                    break

        # Insert config cell
        nb['cells'].insert(insert_index, config_cell)

        # Save notebook
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)

        print(f"  [OK] Added configuration cell")
        print(f"    - EPOCHS: {config['epochs']}")
        print(f"    - BATCH_SIZE: {config['batch']}")
        print(f"    - LEARNING_RATE: {config['lr']}")
        print(f"    - IMAGE_SIZE: {config['img_size']}")

    except Exception as e:
        print(f"  [ERROR] {e}")

print("\n" + "="*60)
print("[OK] UPDATE COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Open any notebook in Colab or Jupyter")
print("2. Find the CONFIGURATION cell (added right after setup)")
print("3. Change EPOCHS, BATCH_SIZE, LEARNING_RATE as needed")
print("4. Run the notebook - uses your configuration!")
print("\nSee TRAINING_CONFIG_GUIDE.md for detailed guidance")
print("="*60 + "\n")
