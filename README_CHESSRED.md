# ChessReD Dataset Integration

This document explains how to download and use the ChessReD (Chess Recognition Dataset) to improve your model's performance.

## About ChessReD

- **Size**: 10,800 real-world images
- **Annotation**: Fully annotated with FEN strings and board corner coordinates
- **Diversity**: Images captured from various angles (0° to 60°) and lighting conditions
- **Source**: **NOT on Hugging Face** - Hosted on:
  - [4TU.ResearchData](https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f)
  - Zenodo (search for "ChessReD" or "Chess Recognition Dataset")
- **License**: CC BY-NC-SA 4.0 (requires account and license acceptance)

## Installation

First, install the required library:

```bash
pip install datasets tqdm
```

Or update your requirements:

```bash
pip install -r requirements.txt
```

## Download the Dataset

### Option 1: Manual Download (Recommended for ChessReD)

ChessReD requires manual download:

1. Visit [4TU.ResearchData](https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f)
2. Create an account and accept the CC BY-NC-SA 4.0 license
3. Download the dataset
4. Extract and organize the files to match the project structure:
   ```
   data/game0_chessred/
   ├── images/
   │   ├── frame_000000.jpg
   │   └── ...
   └── game0_chessred.csv  (with columns: from_frame, fen)
   ```

### Option 2: Alternative Dataset (Automatic)

The script will try to use alternative chess datasets from Hugging Face:

```bash
python download_chessred.py --max_images 100
```

This will attempt to download from `dopaul/chess-pieces-merged` or other available datasets.

## Usage in Training

Once downloaded, the dataset will be automatically included when you run training:

```bash
python train.py --data_root ./data --epochs 15
```

The `load_all_games` function will automatically detect the `game0_chessred` folder and include it in training.

## Benefits

1. **Fixes Overfitting**: Your current dataset has ~700 images. Adding 10,800 diverse images will significantly reduce the 4.49% overfitting gap noted in MODEL_ANALYSIS.md.

2. **Diverse Perspectives**: ChessReD includes angles from 0° to 60°, making your perspective transformation more robust.

3. **Real-World Conditions**: Unlike synthetic data, these are actual photos with varying lighting, backgrounds, and board conditions.

4. **Corner Labels**: The dataset includes board corner coordinates, which you can use to validate/improve your board detection algorithm.

## Custom Options

You can customize the download location and folder name:

```bash
python download_chessred.py --output_root ./data --game_name game0_chessred
```

## Dataset Structure

After download, your data structure will look like:

```
data/
├── game0_chessred/
│   ├── images/
│   │   ├── frame_000000.jpg
│   │   ├── frame_000001.jpg
│   │   └── ...
│   └── game0_chessred.csv
├── game6_per_frame/
│   └── ...
└── ...
```

## Notes

- The download may take some time depending on your internet connection (dataset is ~several GB)
- Images are saved as JPG format to save space
- The CSV includes metadata about which split (train/val/test) each image came from
- Corner coordinates are included in the CSV if available
