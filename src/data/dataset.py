"""
PyTorch Dataset for chessboard square classification.
Handles loading images and converting FEN strings to labels.
"""

import numpy as np
from typing import Optional, Callable

# PyTorch and image processing imports
# These are optional - code handles missing dependencies gracefully
try:
    import torch  # type: ignore
    from torch.utils.data import Dataset  # type: ignore
    from PIL import Image  # type: ignore
    import pandas as pd  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # Placeholder for type hints

# Import FEN parsing functions from data_loader
from .data_loader import fen_to_labels


class ChessboardDataset(Dataset):
    """
    PyTorch Dataset for chessboard square classification.
    
    This dataset loads chessboard images and their corresponding FEN strings,
    converting them to label grids for training.
    
    Note: This is the basic version. Square extraction and board detection
    will be added in subsequent implementations.
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        transform: Optional[Callable] = None,
        flatten_labels: bool = False
    ):
        """
        Args:
            df: Pandas DataFrame with required columns:
                - 'image_path': Path to the chessboard image
                - 'fen': FEN string representing the board state
            transform: Optional PyTorch transforms for image preprocessing
            flatten_labels: If True, labels are flattened to shape (64,).
                          If False, labels are 2D grid of shape (8, 8).
                          Default: False (2D format for per-square classification)
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for ChessboardDataset. "
                "Install with: pip install torch torchvision"
            )
        
        self.df = df
        self.transform = transform
        self.flatten_labels = flatten_labels
        
        # Validate required columns
        required_cols = ['image_path', 'fen']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"DataFrame missing required columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )
    
    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int):
        """
        Returns a single sample (image, label) from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (image_tensor, label_tensor):
            - image_tensor: Preprocessed image as torch.Tensor
            - label_tensor: Board state labels as torch.LongTensor
                           Shape: (8, 8) if flatten_labels=False, (64,) if True
        """
        # 1. Load Image
        img_path = self.df.iloc[idx]['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Could not load image at {img_path}: {e}")
        
        # 2. Get Labels using our FEN parser
        fen = self.df.iloc[idx]['fen']
        label_grid = fen_to_labels(fen, flatten=self.flatten_labels)
        
        # 3. Apply transforms to image
        if self.transform:
            image = self.transform(image)
        else:
            # Default: Convert PIL Image to tensor
            from torchvision import transforms  # type: ignore
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
        
        # 4. Convert label to tensor (LongTensor for CrossEntropyLoss)
        label_tensor = torch.tensor(label_grid, dtype=torch.long)
        
        return image, label_tensor


def load_game_data(
    data_root: str,
    game_numbers: list,
    images_folder: str = "tagged_images"
) -> pd.DataFrame:
    """
    Loads chess game data from the actual data structure.
    
    Data structure expected:
    data_root/
        game2_per_frame/
            game2.csv
            tagged_images/
                frame_000200.jpg
                frame_000588.jpg
                ...
        game4_per_frame/
            game4.csv
            tagged_images/
                ...
    
    Args:
        data_root: Root directory containing game folders
                  (e.g., "/Users/rodrigo/Downloads/Project 1,2,3 - Labeled Chess data (PGN games will be added later)-20251231")
        game_numbers: List of game numbers to load (e.g., [2, 4, 5, 6, 7])
        images_folder: Name of the images subfolder (default: "tagged_images")
    
    Returns:
        DataFrame with columns: 'image_path', 'fen', 'game_number', 'frame_number'
    """
    if not TORCH_AVAILABLE:
        raise ImportError("pandas is required. Install with: pip install pandas")
    
    import os
    
    all_data = []
    
    for game_num in game_numbers:
        game_folder = os.path.join(data_root, f"game{game_num}_per_frame")
        csv_path = os.path.join(game_folder, f"game{game_num}.csv")
        images_dir = os.path.join(game_folder, images_folder)
        
        # Check if paths exist
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found: {csv_path}")
            continue
        if not os.path.exists(images_dir):
            print(f"Warning: Images directory not found: {images_dir}")
            continue
        
        # Read CSV
        df_game = pd.read_csv(csv_path)
        
        # Validate CSV columns
        if 'from_frame' not in df_game.columns or 'fen' not in df_game.columns:
            print(f"Warning: CSV missing required columns. Found: {df_game.columns}")
            continue
        
        # Create image paths and filter for existing images
        for idx, row in df_game.iterrows():
            frame_num = row['from_frame']
            # Format frame number to match image filename (e.g., 200 -> frame_000200.jpg)
            image_filename = f"frame_{frame_num:06d}.jpg"
            image_path = os.path.join(images_dir, image_filename)
            
            # Only add if image exists
            if os.path.exists(image_path):
                all_data.append({
                    'image_path': image_path,
                    'fen': row['fen'],
                    'game_number': game_num,
                    'frame_number': frame_num
                })
            else:
                print(f"Warning: Image not found: {image_path}")
    
    if not all_data:
        raise ValueError(
            f"No valid data found. Check that:\n"
            f"1. Data root exists: {data_root}\n"
            f"2. Game folders exist: game{game_numbers[0]}_per_frame, etc.\n"
            f"3. CSV files and images are present"
        )
    
    return pd.DataFrame(all_data)


def create_sample_dataframe(num_samples: int = 10) -> pd.DataFrame:
    """
    Creates a sample DataFrame with placeholder data for testing.
    
    Args:
        num_samples: Number of placeholder samples to create
        
    Returns:
        DataFrame with 'image_path' and 'fen' columns
    """
    if not TORCH_AVAILABLE:
        raise ImportError("pandas is required. Install with: pip install pandas")
    
    # Placeholder FEN strings (standard starting position)
    placeholder_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    
    # Placeholder image paths
    data = {
        'image_path': [f"data/game1/images/frame_{i:04d}.jpg" for i in range(num_samples)],
        'fen': [placeholder_fen] * num_samples
    }
    
    return pd.DataFrame(data)


# Example usage
if __name__ == "__main__":
    import os
    
    # Path to the actual data
    DATA_ROOT = "/Users/rodrigo/Downloads/Project 1,2,3 - Labeled Chess data (PGN games will be added later)-20251231"
    
    print("=" * 60)
    print("Chessboard Dataset Example")
    print("=" * 60)
    
    # Try loading actual data
    if os.path.exists(DATA_ROOT):
        print(f"\nLoading data from: {DATA_ROOT}")
        try:
            # Load games 2, 4, 5, 6, 7
            df = load_game_data(DATA_ROOT, game_numbers=[2, 4, 5, 6, 7])
            print(f"\nLoaded {len(df)} samples from {df['game_number'].nunique()} games")
            print(f"\nGames: {sorted(df['game_number'].unique())}")
            print(f"\nSample rows:")
            print(df.head())
            
            # Create dataset
            from torchvision import transforms  # type: ignore
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            dataset = ChessboardDataset(df, transform=transform)
            print(f"\nDataset created with {len(dataset)} samples")
            print(f"Dataset class: {type(dataset).__name__}")
            
            # Test loading one sample
            if len(dataset) > 0:
                image, label = dataset[0]
                print(f"\nSample 0:")
                print(f"  Image shape: {image.shape}")
                print(f"  Label shape: {label.shape}")
                print(f"  Label dtype: {label.dtype}")
            
        except Exception as e:
            print(f"\nError loading data: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nData root not found: {DATA_ROOT}")
        print("Using placeholder data instead...")
        
        # Create sample DataFrame
        df = create_sample_dataframe(num_samples=5)
        print("\nSample DataFrame:")
        print(df)
        
        print("\nNote: To use actual data, update DATA_ROOT path in this file.")

