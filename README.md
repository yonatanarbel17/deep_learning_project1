# Chess Board State Recognition from Real-World Images

**Authors:** Yonatan Arbel, Guy Shick, Tomer Tocker

## Overview

A deep learning system that classifies each of the 64 chessboard squares from real-world images into 14 classes (6 white pieces, 6 black pieces, empty, occluded) and reconstructs the full board state in FEN notation. Achieves **95.37% validation accuracy** using an EfficientNet-B2 backbone with gradual unfreezing.

## Environment Setup

**Python version:** 3.10+ recommended (tested on 3.10 and 3.12)

```bash
git clone https://github.com/yonatanarbel17/deep_learning_project1.git
cd deep_learning_project1

# Create virtual environment (choose one):
# Option A: venv
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Option B: conda
conda create -n chess python=3.10
conda activate chess

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, CUDA GPU recommended for training (inference works on CPU). The `timm` library is required for the EfficientNet-B2 backbone.

## Training

**Data format for training:** The training script uses the original data format — each game in its own folder under `data/`. The data is already included in the repository in the correct structure:

```
data/
├── game2_per_frame/          # Original game frames
│   ├── tagged_images/        # Frame images (frame_000200.jpg, ...)
│   └── game2.csv             # FEN labels + occlusion annotations
├── game2_per_frame_bright/   # Augmented variant (brightness)
├── game2_per_frame_dark/     # Augmented variant (darkened)
├── game2_per_frame_color/    # Augmented variant (color-shifted)
├── game2_per_frame_noisy/    # Augmented variant (noise-injected)
├── game4_per_frame/
├── ...
└── game12_per_frame_noisy/
```

> **Note:** The file `gt.csv` in the repository root contains the same data in the submission format required by the course (columns: image_name, FEN, view specification). Training uses the per-game folder format above.

Train the model from scratch:

```bash
python train.py --data_root data --epochs 30
```

Key training arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | (required) | Path to data directory containing game folders |
| `--epochs` | 30 | Number of training epochs |
| `--backbone` | efficientnet_b2 | Model backbone |
| `--batch_size` | 4 | Batch size (boards) |
| `--lr` | 5e-5 | Initial learning rate |

The training script will:
1. Load all game data and split by frame groups (80/20 train/val)
2. Train with gradual unfreezing (60% frozen epochs 1-3, full fine-tune from epoch 4)
3. Save the best model to `outputs/best_model.pth`
4. Generate training curves, per-class accuracy, and confusion matrix
5. Optimize OOD threshold and calibrate temperature

### Training on Google Colab

```bash
!git clone https://github.com/yonatanarbel17/deep_learning_project1.git
%cd deep_learning_project1
!pip install timm
!python train.py --data_root data --epochs 30
```

## Demo (Quick Inference)

Run the model on a single chess board image:

```bash
# From an image file
python demo.py --image path/to/board_image.jpg

# From a game frame
python demo.py --game_dir data/game2_per_frame --frame 200

# With custom output directory
python demo.py --image path/to/board.jpg --output_dir results
```

This will print the predicted FEN string and save a visualization (original image + predicted board) to the `results/` directory.

## Inference / Evaluation

Evaluate the trained model on specific game(s):

```bash
python evaluate_game.py --data_root data --games 2
```

Evaluate on original frames only (no augmented variants):

```bash
python evaluate_game.py --data_root data --games 2 --original_only
```

Arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | data | Path to data directory |
| `--model` | outputs/best_model.pth | Path to trained model weights |
| `--games` | (required) | Comma-separated game numbers (e.g., "2" or "2,4") |
| `--original_only` | false | Only evaluate on original (non-augmented) frames |

## Occlusion Tagging Tools

Tag occluded squares interactively:
```bash
python tag_occlusions.py data/game5_per_frame
```

Propagate tags to augmented variants:
```bash
python propagate_occlusions.py
```

## Project Structure

```
├── train.py                  # Main training script
├── predict.py                # Evaluation API: predict_board(image) → 8x8 tensor
├── demo.py                   # Quick inference on a single image
├── evaluate_game.py          # Evaluate model on specific game(s)
├── tag_occlusions.py         # Interactive occlusion annotation tool
├── propagate_occlusions.py   # Propagate tags to augmented variants
├── gt.csv                    # Ground truth in submission format
├── past_architectures.md     # Training run history (Runs 1-4)
├── report.pdf                # Project report
├── src/
│   ├── data/
│   │   ├── data_loader.py    # FEN parsing, label mapping
│   │   ├── dataset.py        # Board-level PyTorch Dataset, transforms, sampler
│   │   └── board_detection.py # Perspective correction
│   ├── models/
│   │   └── classifier.py     # EfficientNet-B2 classifier
│   ├── inference/
│   │   └── predictor.py      # Prediction with temperature scaling and OOD detection
│   ├── training/
│   │   └── trainer.py        # Training loop with gradual unfreezing
│   └── utils/
│       └── visualization.py  # Training curves and reports
├── scripts/
│   └── plot_run4_curves.py   # Plot training curves for Run 4
├── data/                     # Training data (10 games with augmented variants)
├── outputs/
│   └── best_model.pth        # Trained EfficientNet-B2 weights
├── requirements.txt
└── pyproject.toml
```

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | 95.37% |
| Training Epochs | 30 |
| Training Data | 10 games, 2,980 boards |
| Classes | 14 (12 pieces + empty + occluded) |
| Backbone | EfficientNet-B2 (pretrained ImageNet) |
