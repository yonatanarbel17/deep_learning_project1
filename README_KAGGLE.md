# Kaggle Chess Positions Dataset Integration

This document explains how to download and use the Kaggle Chess Positions dataset to improve your model's performance.

## About the Dataset

- **Source**: [Kaggle - koryakinp/chess-positions](https://www.kaggle.com/datasets/koryakinp/chess-positions)
- **Type**: Synthetic chess board images
- **Generator**: Created using [chess-generator](https://github.com/koryakinp/chess-generator)
- **Annotation**: Includes FEN strings for board positions

## Prerequisites

1. **Kaggle Account**: You need a Kaggle account to download datasets
2. **Kaggle API Setup**: 
   - Go to your Kaggle account settings
   - Scroll to "API" section
   - Click "Create New Token" to download `kaggle.json`
   - Place it in `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables)

## Installation

Install the required library:

```bash
pip install kagglehub
```

Or update your requirements:

```bash
pip install -r requirements.txt
```

## Download the Dataset

Run the download script:

```bash
python download_kaggle_chess.py
```

This will:
1. Download the dataset from Kaggle using kagglehub
2. Process all images and extract FEN annotations
3. Save them to `data/game1_kaggle/images/`
4. Create a CSV file at `data/game1_kaggle/game1_kaggle.csv`
5. Format everything to match your existing data structure

## Usage in Training

Once downloaded, the dataset will be automatically included when you run training:

```bash
python train.py --data_root ./data --epochs 15
```

The `load_all_games` function will automatically detect the `game1_kaggle` folder and include it in training.

## Benefits

1. **Synthetic Data**: High-quality rendered images with perfect annotations
2. **Diverse Positions**: Generated from real chess games, providing varied board states
3. **Perfect Labels**: FEN strings are mathematically correct (no human labeling errors)
4. **Complementary to Real Data**: Works well alongside real-world datasets like ChessReD

## Custom Options

You can customize the download location and folder name:

```bash
python download_kaggle_chess.py --output_root ./data --game_name game1_kaggle
```

## Dataset Structure

After download, your data structure will look like:

```
data/
├── game1_kaggle/
│   ├── images/
│   │   ├── frame_000000.jpg
│   │   ├── frame_000001.jpg
│   │   └── ...
│   └── game1_kaggle.csv
├── game0_chessred/
│   └── ...
└── ...
```

## Notes

- The script automatically converts images to JPG format for consistency
- FEN annotations are extracted from JSON files if available
- If JSON files aren't found, the script attempts to extract FEN from filenames
- Images without FEN annotations are skipped
- The download location is managed by kagglehub (usually in `~/.cache/kagglehub/`)

## Troubleshooting

**Error: "kagglehub library not installed"**
- Run: `pip install kagglehub`

**Error: "Failed to download dataset"**
- Check your Kaggle API credentials
- Ensure you have internet connection
- Verify the dataset is publicly available

**No images processed**
- Check if the dataset structure matches expectations
- Inspect the downloaded dataset folder manually
- The script prints the dataset path for inspection
