"""
PyTorch Dataset for chessboard square classification.
Handles loading images, extracting 64 squares, and converting FEN strings to labels.
"""

import numpy as np
from typing import Optional, Callable, Tuple, List
import os

# PyTorch and image processing imports
import torch  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore
from PIL import Image  # type: ignore
import pandas as pd  # type: ignore

# Computer vision imports for board rectification
try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Import FEN parsing functions from data_loader
from .data_loader import fen_to_labels


def extract_squares(image: Image.Image, board_size: int = 400) -> List[Image.Image]:
    """
    Extracts 64 individual squares from a chessboard image.
    
    This function divides the board image into an 8x8 grid and returns
    each square as a separate image. For now, assumes the board fills
    the image. Later, this will be enhanced with board detection.
    
    Args:
        image: PIL Image of the chessboard
        board_size: Size to resize the board to before extraction (default: 400)
                   Each square will be board_size/8 pixels
    
    Returns:
        List of 64 PIL Images, one for each square.
        Order: rank 8 (top) to rank 1 (bottom), left to right
        Square at index (row*8 + col) corresponds to rank (8-row), file (col+1)
    """
    # Resize image to board_size x board_size for consistent square sizes
    image = image.resize((board_size, board_size), Image.LANCZOS)
    
    squares = []
    square_size = board_size // 8
    
    # Extract squares row by row (rank 8 to rank 1)
    for row in range(8):
        for col in range(8):
            # Calculate bounding box for this square
            left = col * square_size
            top = row * square_size
            right = (col + 1) * square_size
            bottom = (row + 1) * square_size
            
            # Crop the square
            square = image.crop((left, top, right, bottom))
            squares.append(square)
    
    return squares


class ChessboardDataset(Dataset):
    """
    PyTorch Dataset for chessboard square classification.
    
    This dataset loads chessboard images, extracts 64 individual squares,
    and converts FEN strings to per-square labels. Each board image becomes
    64 training samples (one per square).
    
    Optimizations:
    - Caching: Avoids reloading the same board image multiple times
    - Perspective transform: Optional board rectification for better square extraction
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        transform: Optional[Callable] = None,
        board_size: int = 400,
        square_size: int = 224
    ):
        """
        Args:
            df: Pandas DataFrame with required columns:
                - 'image_path': Path to the chessboard image
                - 'fen': FEN string representing the board state
            transform: Optional PyTorch transforms for square preprocessing
            board_size: Size to resize board to before square extraction (default: 400)
            square_size: Size to resize each square to (default: 224 for ResNet)
        """
        self.df = df
        self.transform = transform
        self.board_size = board_size
        self.square_size = square_size
        
        # Validate required columns
        required_cols = ['image_path', 'fen']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"DataFrame missing required columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Pre-expand: Each board becomes 64 square samples
        self.samples = self._prepare_samples()
        
        # Cache to avoid reloading the same image multiple times
        # (Since we extract 64 squares from each board)
        self.last_image_path = None
        self.last_squares = None
    
    def _prepare_samples(self) -> List[dict]:
        """
        Expands each board into 64 square samples.
        Returns list of samples, each with image_path, fen, row, col.
        """
        samples = []
        for idx, row in self.df.iterrows():
            fen = row['fen']
            label_grid = fen_to_labels(fen)  # Shape: (8, 8)
            
            # Create 64 samples (one per square)
            for rank_idx in range(8):  # rank 8 (top) to rank 1 (bottom)
                for file_idx in range(8):  # file a (left) to file h (right)
                    square_label = label_grid[rank_idx, file_idx]
                    samples.append({
                        'image_path': row['image_path'],
                        'fen': fen,
                        'rank': rank_idx,  # 0-7 (rank 8 to rank 1)
                        'file': file_idx,  # 0-7 (file a to file h)
                        'label': square_label
                    })
        
        return samples
    
    def _apply_perspective_transform(
        self, 
        image_cv: np.ndarray, 
        corners: np.ndarray
    ) -> Image.Image:
        """
        Applies perspective transform to rectify the board to a top-down view.
        
        This function warps a skewed/rotated board image into a canonical
        square board. The corners should be detected automatically or provided
        in the CSV (future enhancement).
        
        Args:
            image_cv: Board image as OpenCV format (BGR numpy array)
            corners: Array of 4 corner points in order:
                    [top-left, top-right, bottom-right, bottom-left]
                    Shape: (4, 2) with float32 coordinates
        
        Returns:
            Rectified board image as PIL Image (RGB)
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for perspective transform. Install with: pip install opencv-python")
        
        # Define destination points (canonical square)
        pts1 = np.float32(corners)
        pts2 = np.float32([
            [0, 0],                           # Top-left
            [self.board_size, 0],             # Top-right
            [self.board_size, self.board_size],  # Bottom-right
            [0, self.board_size]              # Bottom-left
        ])
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        
        # Apply warp
        warped = cv2.warpPerspective(image_cv, matrix, (self.board_size, self.board_size))
        
        # Convert back to PIL Image (BGR -> RGB)
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        return Image.fromarray(warped_rgb)
    
    def __len__(self) -> int:
        """Returns the number of square samples (64 per board)."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a single square sample (square_image, square_label).
        
        Args:
            idx: Index of the square sample to retrieve
            
        Returns:
            Tuple of (square_tensor, label_tensor):
            - square_tensor: Preprocessed square image as torch.Tensor of shape (3, square_size, square_size)
            - label_tensor: Square label as torch.LongTensor (single class ID 0-12)
        """
        sample = self.samples[idx]
        img_path = sample['image_path']
        
        # Optimization: If this is the same image as the previous sample,
        # reuse the cached squares instead of reloading from disk
        if img_path != self.last_image_path:
            # 1. Load full board image
            try:
                board_image = Image.open(img_path).convert('RGB')
            except Exception as e:
                raise FileNotFoundError(f"Could not load image at {img_path}: {e}")
            
            # Optional: Apply perspective transform if corners are available
            # This would be used when board detection is implemented
            # if 'corners' in sample:
            #     board_image_cv = cv2.cvtColor(np.array(board_image), cv2.COLOR_RGB2BGR)
            #     board_image = self._apply_perspective_transform(board_image_cv, sample['corners'])
            
            # 2. Extract all 64 squares from the board
            squares = extract_squares(board_image, board_size=self.board_size)
            
            # Cache the squares and image path
            self.last_squares = squares
            self.last_image_path = img_path
        else:
            # Reuse cached squares
            squares = self.last_squares
        
        # 3. Get the specific square we need (rank*8 + file)
        square_idx = sample['rank'] * 8 + sample['file']
        square_image = squares[square_idx]
        
        # 4. Resize square to target size
        square_image = square_image.resize((self.square_size, self.square_size), Image.LANCZOS)
        
        # 5. Apply transforms to square
        if self.transform:
            square_tensor = self.transform(square_image)
        else:
            # Default: Convert PIL Image to tensor
            from torchvision import transforms  # type: ignore
            to_tensor = transforms.ToTensor()
            square_tensor = to_tensor(square_image)
        
        # 6. Convert label to tensor (single class ID)
        label_tensor = torch.tensor(sample['label'], dtype=torch.long)
        
        return square_tensor, label_tensor


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


def get_default_transforms(
    square_size: int = 224,
    is_training: bool = True
) -> Callable:
    """
    Returns default transforms for square images.
    
    Args:
        square_size: Target size for each square (default: 224 for ResNet)
        is_training: If True, includes data augmentation. If False, only normalization.
    
    Returns:
        PyTorch transform pipeline
    """
    from torchvision import transforms  # type: ignore
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((square_size, square_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(5),  # Small rotation for robustness
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        return transforms.Compose([
            transforms.Resize((square_size, square_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 4,
    square_size: int = 224,
    board_size: int = 400
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and validation DataLoaders with proper transforms.
    
    Args:
        train_df: DataFrame for training data
        val_df: DataFrame for validation data
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes for data loading
        square_size: Size to resize each square to
        board_size: Size to resize board to before square extraction
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create transforms
    train_transform = get_default_transforms(square_size=square_size, is_training=True)
    val_transform = get_default_transforms(square_size=square_size, is_training=False)
    
    # Create datasets
    train_dataset = ChessboardDataset(
        df=train_df,
        transform=train_transform,
        board_size=board_size,
        square_size=square_size
    )
    
    val_dataset = ChessboardDataset(
        df=val_df,
        transform=val_transform,
        board_size=board_size,
        square_size=square_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Critical: shuffle to break temporal correlation
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_sample_dataframe(num_samples: int = 10) -> pd.DataFrame:
    """
    Creates a sample DataFrame with placeholder data for testing.
    
    Args:
        num_samples: Number of placeholder samples to create
        
    Returns:
        DataFrame with 'image_path' and 'fen' columns
    """
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
            
            # Split data by games for train/val
            train_games = [2, 4, 5]
            val_games = [6, 7]
            
            train_df = df[df['game_number'].isin(train_games)]
            val_df = df[df['game_number'].isin(val_games)]
            
            print(f"\nTrain: {len(train_df)} boards from games {train_games}")
            print(f"Val: {len(val_df)} boards from games {val_games}")
            
            # Create dataloaders (automatically extracts 64 squares per board)
            train_loader, val_loader = create_dataloaders(
                train_df=train_df,
                val_df=val_df,
                batch_size=32,
                square_size=224,
                board_size=400
            )
            
            print(f"\nTrain loader: {len(train_loader.dataset)} square samples")
            print(f"Val loader: {len(val_loader.dataset)} square samples")
            print(f"  (Each board = 64 squares)")
            
            # Test loading one batch
            if len(train_loader) > 0:
                square_images, square_labels = next(iter(train_loader))
                print(f"\nBatch sample:")
                print(f"  Square images shape: {square_images.shape}")  # (batch, 3, 224, 224)
                print(f"  Square labels shape: {square_labels.shape}")  # (batch,)
                print(f"  Number of classes in batch: {len(torch.unique(square_labels))}")
            
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

