# Chess Board State Recognition from Real-World Images

**Authors:** Yonatan Arbel, Guy Shick, Tomer Tocker

## Overview

A deep learning system that classifies each of the 64 chessboard squares from real-world images into 14 classes (6 white pieces, 6 black pieces, empty, occluded) and reconstructs the full board state in FEN notation. Achieves **95.37% validation accuracy** using an EfficientNet-B2 backbone with gradual unfreezing.

## Environment Setup

```bash
git clone https://github.com/yonatanarbel17/deep_learning_project1.git
cd deep_learning_project1
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, CUDA GPU recommended. The `timm` library is required for the EfficientNet-B2 backbone.

## Training

Train the model from scratch:

```bash
python train.py --data_root data --epochs 50
```

Key training arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | (required) | Path to data directory containing game folders |
| `--epochs` | 50 | Number of training epochs |
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
!python train.py --data_root data --epochs 50
```

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
python tag_occlusions.py --game_dir data/game5_per_frame
```

Propagate tags to augmented variants:
```bash
python propagate_occlusions.py --data_root data
```

## Project Structure

```
├── train.py                  # Main training script
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
| Training Data | 10 games, 2,980 boards |
| Classes | 14 (12 pieces + empty + occluded) |
| Backbone | EfficientNet-B2 (pretrained ImageNet) |
