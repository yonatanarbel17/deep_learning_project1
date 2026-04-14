# Chess Board State Recognition from Real-World Images

**Authors:** Yonatan Arbel, Guy Shick, Tomer Tocker

## Overview

A deep learning system that classifies each of the 64 chessboard squares from real-world images into 14 classes (6 white pieces, 6 black pieces, empty, occluded) and reconstructs the full board state in FEN notation. Achieves **95.37% validation accuracy** using an EfficientNet-B2 backbone with gradual unfreezing.

## Architecture

- **Backbone:** EfficientNet-B2 (pretrained on ImageNet)
- **Classifier:** Linear head mapping to 14 classes (P, N, B, R, Q, K, p, n, b, r, q, k, empty, occluded)
- **Board Detection:** OpenCV perspective correction to canonical top-down view
- **Square Extraction:** 64 squares per board with 70% overlap padding for context
- **OOD Detection:** Hybrid supervised occlusion class + confidence thresholding
- **Post-Processing:** Chess rule constraints (no pawns on ranks 1/8, one king per side, piece count limits)
- **Calibration:** Temperature scaling (T=1.24) for confidence calibration

## Training Strategy

- **Gradual Unfreezing:** 60% backbone frozen for epochs 1-3, full fine-tuning from epoch 4 at halved LR
- **Optimizer:** AdamW (LR=5e-5, weight_decay=1e-3)
- **Loss:** CrossEntropy with label smoothing (0.1) and inverse-frequency class weights
- **Scheduler:** ReduceLROnPlateau (patience=3, factor=0.5)
- **Augmentations:** ColorJitter, random affine, Gaussian blur, random erasing, random grayscale
- **Offline augmentations:** Bright, dark, color, and noisy variants per game
- **Split:** Random 80/20 by (game_id, frame_number) groups to prevent augmentation leakage

## Project Structure

```
├── train.py                  # Main training script
├── evaluate_game.py          # Evaluate model on specific game(s)
├── tag_occlusions.py         # Interactive occlusion annotation tool
├── propagate_occlusions.py   # Propagate occlusion tags to augmented variants
├── past_architectures.md     # Training run history (Runs 1-4)
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
│   └── best_model.pth        # Trained EfficientNet-B2 weights (95.37% val acc)
├── requirements.txt
└── pyproject.toml
```

## Usage

### Training (Google Colab)

```bash
git clone https://github.com/yonatanarbel17/deep_learning_project1.git
cd deep_learning_project1
pip install timm
python train.py --data_root data --epochs 50
```

### Evaluate on a specific game

```bash
python evaluate_game.py --data_root data --games 2
```

### Occlusion Tagging

Tag occluded squares interactively:
```bash
python tag_occlusions.py --game_dir data/game5_per_frame
```

Propagate tags to augmented variants:
```bash
python propagate_occlusions.py --data_root data
```

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy (random frame-group split) | 95.37% |
| Game 2 Accuracy (held-out evaluation) | 99.35% |
| Number of Classes | 14 |
| Training Data | 10 games, 2980 boards |

## Training Evolution

| Run | Architecture | Val Accuracy | Key Change |
|-----|-------------|-------------|------------|
| 1 | EfficientNet-B2 (full fine-tune) | Overfitting | Game-level split, no freezing |
| 2 | EfficientNet-B2 (frozen + anti-overfit) | 91.37% | Freeze layers, lower LR, frame-group split |
| 3 | EfficientNet-B2 (gradual unfreezing) | 94.97% | Unfreeze at epoch 4, 25 epochs |
| 4 | EfficientNet-B2 (50 epochs) | 95.37% | Extended training to 50 epochs |
