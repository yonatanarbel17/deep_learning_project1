# Chessboard Square Classification and Board-State Reconstruction

## Project Overview

This project implements a deep learning system to classify each of the 64 chessboard squares from real-world images into piece classes (white pawn, black knight, etc.) and reconstruct the complete board state in FEN (Forsyth-Edwards Notation) format.

### Problem Statement
Given a single static image of a real chessboard, the system must:
- Classify each square into one of 13 classes (12 piece types × 2 colors + empty)
- Handle occlusions (hands, heads, etc.) by outputting "unknown" for occluded squares
- Reconstruct the board state as a FEN string
- Generate a digital chess diagram from the FEN
- Generalize to new games with different lighting, boards, and camera angles

### Key Constraints
- **No temporal inference**: The classifier must work on a single static image only
- **Occlusion handling**: Must detect and mark occluded squares as "unknown"
- **Robustness**: Must work on new games not seen during training
- **Data leakage prevention**: Training/validation split must be by game, not by frame

## Data Format

Each game is provided as a ZIP file containing:
- `images/` folder: Frame images (e.g., `frame_130.jpg`, `frame_162.jpg`)
- `game.csv` file: Contains mappings between frames and board states
  - `from_frame` / `to_frame`: Frame number (both identical for each distinct board state)
  - `fen`: FEN string representing the board state at that frame

**Note**: The CSV only labels distinct board states. Intermediate frames (with occlusions or during piece movement) are not explicitly labeled but can be used for training the "unknown" class.

## Architecture

The system uses a modular pipeline approach:

1. **Spatial Transformer Network (STN)**: Rectifies and warps the board image to a canonical top-down view
   - Handles perspective distortion and rotation
   - Ensures consistent square positioning

2. **Square Extraction**: Divides the rectified board into 64 individual square images
   - Each square is processed independently
   - Enables treating each square as a separate training sample

3. **Out-of-Distribution (OOD) Detection**: Identifies occluded or uncertain squares
   - **Autoencoder-based**: Trained only on clean squares; high reconstruction error indicates occlusion
   - **Entropy-based**: High entropy in classifier output indicates uncertainty
   - Squares flagged as OOD are marked as "unknown"

4. **ResNet-18 Classifier**: Performs 13-class classification
   - 12 piece classes (6 white + 6 black pieces)
   - 1 empty square class
   - Uses pre-trained ImageNet weights for transfer learning

5. **FEN Reconstruction**: Combines 64 square predictions into FEN notation
   - Handles "unknown" squares appropriately
   - Generates digital chess diagram using `python-chess`

## Project Structure

```
DL_project/
├── data/                    # Data directory (extract game ZIP files here)
│   ├── game1/
│   │   ├── images/
│   │   └── game.csv
│   ├── game2/
│   │   ├── images/
│   │   └── game.csv
│   └── ...
├── src/
│   ├── data/               # Data loading utilities
│   │   └── data_loader.py  # FEN parsing and data loading utilities
│   ├── models/             # Model architectures
│   │   ├── stn.py          # Spatial Transformer Network
│   │   ├── autoencoder.py  # OOD detection via reconstruction
│   │   ├── classifier.py   # ResNet-18 classifier
│   │   └── pipeline.py     # Combined model pipeline
│   ├── utils/              # Utility functions
│   │   ├── fen_utils.py    # FEN generation and parsing
│   │   └── visualization.py # Chess diagram generation
│   └── training/           # Training scripts
│       └── train.py         # Main training script
├── configs/                # Configuration files
│   └── config.yaml         # Training and model hyperparameters
├── checkpoints/            # Saved model checkpoints
├── outputs/                # Output predictions and diagrams
├── logs/                   # Training logs and TensorBoard files
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

### Installation

1. **Extract data**: Extract game ZIP files into the `data/` directory:
   ```bash
   data/
   ├── game1/
   │   ├── images/
   │   └── game.csv
   ├── game2/
   │   ├── images/
   │   └── game.csv
   └── ...
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure**: Edit `configs/config.yaml` to set:
   - Which games to use for training/validation
   - Model hyperparameters
   - Training settings

## Usage

### Training

Train the model with game-based data splitting:
```bash
python src/training/train.py --config configs/config.yaml
```

The training script will:
- Load games specified in config (e.g., games 1-4 for training, game 5 for validation)
- Extract squares from board images
- Train the classifier with proper data augmentation
- Save checkpoints and training metrics

### Inference

Run inference on a single chessboard image:
```bash
python src/inference.py --image path/to/board.jpg --checkpoint checkpoints/best_model.pth
```

The inference script will:
- Load the trained model
- Process the input image through the full pipeline
- Output the FEN string and generate a chess diagram

## Key Features

- **Game-based data splitting**: Prevents data leakage by ensuring frames from the same game are only in train OR validation
- **Automatic square extraction**: Converts board images into 64 individual square samples
- **Occlusion detection**: Uses Autoencoder reconstruction error and entropy-based uncertainty to identify occluded squares
- **Robust classification**: ResNet-18 with transfer learning for accurate piece recognition
- **FEN generation**: Converts 64 square predictions into standard FEN notation
- **Chess diagram visualization**: Generates digital board diagrams from FEN strings

## Implementation Details

### Data Loading
- Each board image is split into 64 squares
- FEN strings are parsed to extract ground truth labels
- Data augmentation (color jitter, rotation) applied during training
- Shuffling breaks temporal correlation between frames

### Model Training
- Cross-entropy loss with LogSumExp trick for numerical stability
- Adam optimizer with learning rate scheduling
- Early stopping based on validation accuracy
- TensorBoard logging for training visualization

### Occlusion Handling
- Autoencoder trained only on clean squares (pieces + empty)
- Reconstruction error threshold determines if square is occluded
- Entropy of classifier output provides additional uncertainty measure
- Squares with high uncertainty marked as "unknown"

## References

- Course materials on Spatial Transformer Networks, Autoencoders, ResNets
- FEN notation specification
- `python-chess` library for board visualization

