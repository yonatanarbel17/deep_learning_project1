# Past Architectures & Training Runs

---

## Run 1 — EfficientNet-B2 (Full Fine-Tune, Game-Level Split)

**Date:** 2026-04-03
**Branch:** shick
**Status:** Killed at epoch 8 (overfitting)

### Architecture
- **Backbone:** EfficientNet-B2 (pretrained ImageNet)
- **Classifier head:** Dropout(0.3) → Linear(1408, 14)
- **Frozen layers:** None (full fine-tune)
- **Input:** 512×512 board → 64 squares with 70% overlap padding → 224×224 per square

### Training Config
- **LR:** 1e-4 (AdamW, weight_decay=1e-3)
- **Batch size:** 4 boards
- **Label smoothing:** 0.1
- **Class weights:** Inverse-frequency
- **WeightedRandomSampler:** Yes (board-level, mean inverse-frequency)
- **Gradient clipping:** max_norm=1.0
- **LR scheduler:** ReduceLROnPlateau (patience=3, factor=0.5)
- **Early stopping:** patience=5

### Data
- **Total samples:** 2980 boards from 10 games
- **Split:** Game-level (last 20% of game IDs → val)
- **Train:** 2780 samples from 8 games
- **Val:** 200 samples from 2 games (games 11, 12 only)
- **Augmentations (offline):** bright/color/dark/noisy variants for all games (2,4-12)
- **Augmentations (online):** ColorJitter(0.3/0.3/0.2/0.05), ±5° rotation, RandomGrayscale(0.1), GaussianBlur(0.2), RandomErasing(0.25)

### Results (killed at epoch 8)
| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| 1     | 66.11%    | 60.77%  | 2.0699     | 2.3107   |
| 2     | 88.21%    | 65.46%  | 1.6203     | 2.2554   |
| 3     | 91.77%    | 68.73%  | 1.5668     | 2.3177   |
| 4     | 93.20%    | 72.01%  | 1.5348     | 2.3870   |
| 5     | 93.11%    | 72.54%  | 1.5248     | 2.3628   |
| 6     | 94.69%    | 70.65%  | 1.5138     | 2.5003   |
| 7     | 95.17%    | 71.63%  | 1.5145     | 2.3937   |

**Best val accuracy:** 72.54% (epoch 5)

### Analysis
- **Severe overfitting:** 23-point train-val gap by epoch 7 (95% vs 72%)
- **Val loss diverging:** Climbed from 2.25 → 2.50 after epoch 2
- **Root causes:**
  - Full fine-tune with 1e-4 LR on small dataset → model memorized training data
  - Val set too small (200 samples) and only from 2 games → not representative
  - No backbone freezing → early layers overfitting to training distribution

---

## Run 2 — EfficientNet-B2 (60% Frozen, Frame-Group Split)

**Date:** 2026-04-03
**Branch:** shick (commit 4d04894)
**Status:** Completed (15 epochs)

### Architecture
- **Backbone:** EfficientNet-B2 (pretrained ImageNet)
- **Classifier head:** Dropout(0.4) → Linear(1408, 14)
- **Frozen layers:** 179/299 backbone parameter groups (60%) — frozen for all 15 epochs
- **Input:** 512×512 board → 64 squares with 70% overlap padding → 224×224 per square

### Training Config
- **LR:** 5e-5 (AdamW, weight_decay=1e-3) — constant, scheduler never triggered
- **Batch size:** 4 boards
- **Label smoothing:** 0.1
- **Class weights:** Inverse-frequency
- **WeightedRandomSampler:** Yes (board-level)
- **Gradient clipping:** max_norm=1.0
- **LR scheduler:** ReduceLROnPlateau (patience=3, factor=0.5) — never activated
- **Early stopping:** patience=5 — never activated
- **Gradual unfreezing:** Not used in this run (code was pushed but not pulled)

### Data
- **Total samples:** 2980 boards from 10 games
- **Split:** Random frame-group split (20% of unique game_id+frame groups → val)
  - Augmented variants stay with their original (no leakage)
- **Train:** 2385 samples from 10 games
- **Val:** 595 samples from 9 games
- **Augmentations (offline):** bright/color/dark/noisy variants for all games (2,4-12)
- **Augmentations (online):** ColorJitter(0.3/0.3/0.2/0.05), ±5° rotation, RandomGrayscale(0.1), GaussianBlur(0.2), RandomErasing(0.25)

### Results (full 15 epochs)
| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| 1     | 34.07%    | 72.02%  | 2.6657     | 2.5357   |
| 2     | 65.46%    | 77.43%  | 2.0793     | 2.3148   |
| 3     | 72.95%    | 81.29%  | 1.8817     | 2.1777   |
| 4     | 76.33%    | 84.60%  | 1.7977     | 2.1042   |
| 5     | 80.59%    | 85.42%  | 1.7321     | 2.0209   |
| 6     | 82.72%    | 87.52%  | 1.6848     | 2.0186   |
| 7     | 83.64%    | 88.56%  | 1.6729     | 1.9889   |
| 8     | 84.62%    | 88.65%  | 1.6380     | 1.9869   |
| 9     | 86.32%    | 88.79%  | 1.6068     | 1.9882   |
| 10    | 86.48%    | 89.72%  | 1.6085     | 1.9694   |
| 11    | 87.82%    | 90.51%  | 1.5874     | 1.9462   |
| 12    | 88.35%    | 90.35%  | 1.5892     | 1.9627   |
| 13    | 88.40%    | 90.97%  | 1.5585     | 1.9425   |
| 14    | 88.96%    | 91.29%  | 1.5620     | 1.9229   |
| 15    | 90.04%    | 91.37%  | 1.5510     | 1.9209   |

**Best val accuracy:** 91.37% (epoch 15)
**Overfit gap:** -1.33% (val higher than train — no overfitting)
**Optimal OOD threshold:** 0.1 (score=0.9137)
**Calibrated temperature:** 1.2559

### Analysis
- **No overfitting:** Val accuracy consistently above train accuracy throughout training
- **Val loss continuously decreasing:** 2.54 → 1.92 (never diverged)
- **Still improving at epoch 15:** Both train and val curves still trending upward — model has not converged
- **Key improvements over Run 1:**
  - Frame-group split gave representative val set (595 samples from 9 games vs 200 from 2)
  - 60% frozen backbone prevented memorization of low-level features
  - Lower LR (5e-5 vs 1e-4) slowed learning to prevent rapid overfitting
  - Higher dropout (0.4 vs 0.3) added regularization
- **OOD threshold at 0.1:** Essentially means "accept all predictions" — the model is confident enough that filtering by threshold reduces coverage too aggressively
- **Temperature 1.2559:** Slightly >1 means the model is slightly overconfident; temperature scaling softens predictions
- **Room for improvement:** Model was still improving at epoch 15 — more epochs or gradual unfreezing could push accuracy higher
