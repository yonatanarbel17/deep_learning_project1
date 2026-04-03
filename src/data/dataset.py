"""
PyTorch Dataset for chessboard square classification (BOARD-LEVEL).
- Loads one image = returns 64 squares in one go.
- Extracts squares with padding (context).
- Optional rotation augmentation with matching label rotation.
- Optional perspective transformation for top-down view.
"""

from __future__ import annotations

import os
import random
from typing import Optional, Callable, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

import torch  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore

from .data_loader import fen_to_labels
from .board_detection import get_chess_board_upper_look


# -------- Class weight computation --------
def compute_class_weights(train_df: pd.DataFrame, num_classes: int = 14) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from training data.

    Args:
        train_df: DataFrame with 'fen' column containing FEN strings
        num_classes: Number of classes (default 14)

    Returns:
        torch.FloatTensor of shape (num_classes,) with normalized inverse-frequency weights
    """
    # Count occurrences of each class
    class_counts = np.zeros(num_classes)

    for _, row in train_df.iterrows():
        fen = row['fen']
        labels = fen_to_labels(fen, flatten=True)  # (64,)
        for label in labels:
            class_counts[int(label)] += 1

    # Compute inverse-frequency weights
    # For zero-count classes, assign weight 1.0
    weights = np.zeros(num_classes)
    total_samples = class_counts.sum()

    for c in range(num_classes):
        if class_counts[c] > 0:
            weights[c] = total_samples / (num_classes * class_counts[c])
        else:
            weights[c] = 1.0

    # Normalize so they sum to num_classes
    weights = weights * num_classes / weights.sum()

    return torch.FloatTensor(weights)


# -----------------------------
# Square extraction with padding
# -----------------------------
def extract_squares_with_padding(
    board: Image.Image,
    board_size: int,
    square_size: int,
    pad_ratio: float = 0.5
) -> List[Image.Image]:
    """
    Cuts 8x8 squares from board (assumed already board_size x board_size).
    Each square includes padding from neighbors to capture pieces that 
    visually "lean" into adjacent squares due to camera angle.
    
    Args:
        board: Input board image (already resized to board_size x board_size)
        board_size: Expected board dimensions
        square_size: Output size for each extracted square
        pad_ratio: How much of adjacent squares to include (0.5 = half of each neighbor)
    """
    edges = np.round(np.linspace(0, board_size, 9)).astype(int)  # 0..board_size
    squares: List[Image.Image] = []

    for r in range(8):
        for c in range(8):
            left = edges[c]
            right = edges[c + 1]
            top = edges[r]
            bottom = edges[r + 1]

            sw = right - left
            sh = bottom - top
            pad_x = int(round(sw * pad_ratio))
            pad_y = int(round(sh * pad_ratio))

            # apply padding & clip
            L = max(0, left - pad_x)
            R = min(board_size, right + pad_x)
            T = max(0, top - pad_y)
            B = min(board_size, bottom + pad_y)

            sq = board.crop((L, T, R, B))
            sq = sq.resize((square_size, square_size), Image.BILINEAR)
            squares.append(sq)

    return squares


def rotate_board_and_labels(
    board: Image.Image, 
    labels_8x8: np.ndarray, 
    k: int
) -> Tuple[Image.Image, np.ndarray]:
    """
    k in {0,1,2,3} rotate CCW by 90*k
    """
    if k == 0:
        return board, labels_8x8
    if k == 1:
        return board.transpose(Image.ROTATE_90), np.rot90(labels_8x8, 1)
    if k == 2:
        return board.transpose(Image.ROTATE_180), np.rot90(labels_8x8, 2)
    if k == 3:
        return board.transpose(Image.ROTATE_270), np.rot90(labels_8x8, 3)
    raise ValueError("k must be 0..3")


# -----------------------------
# Dataset (BOARD-LEVEL)
# -----------------------------
class ChessboardDataset(Dataset):
    """
    Each item returns:
      - squares_tensor: (64, 3, H, W)
      - labels_tensor:  (64,)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
        board_size: int = 512,
        square_size: int = 224,
        pad_ratio: float = 0.5,
        rotation_augment: bool = True,
        use_perspective_transform: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.board_size = board_size
        self.square_size = square_size
        self.pad_ratio = pad_ratio
        self.rotation_augment = rotation_augment
        self.use_perspective_transform = use_perspective_transform

        required_cols = ["image_path", "fen"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame missing required column '{col}'. Has: {list(self.df.columns)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        fen = row["fen"]

        # 1) Load image
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        board_img = Image.open(img_path).convert("RGB")

        # 2) Apply perspective transformation to get top-down view
        # This detects board corners and warps to perfect square
        board_resized = get_chess_board_upper_look(
            board_img,
            output_size=self.board_size,
            use_perspective=self.use_perspective_transform
        )

        # 3) Labels from FEN (8x8)
        labels_8x8 = fen_to_labels(fen, flatten=False)  # (8,8)

        # 4) Rotation augmentation (0/90/180/270) WITH label rotation
        if self.rotation_augment:
            k = random.randint(0, 3)
            board_resized, labels_8x8 = rotate_board_and_labels(board_resized, labels_8x8, k)

        # 5) Extract 64 squares with padding
        squares = extract_squares_with_padding(
            board_resized,
            board_size=self.board_size,
            square_size=self.square_size,
            pad_ratio=self.pad_ratio
        )

        # 6) Apply transform per-square, stack -> (64,3,H,W)
        if self.transform is None:
            from torchvision import transforms  # type: ignore
            self.transform = transforms.ToTensor()

        sq_tensors = []
        for sq in squares:
            sq_tensors.append(self.transform(sq))
        squares_tensor = torch.stack(sq_tensors, dim=0)

        # 7) Labels -> (64,)
        labels_tensor = torch.tensor(labels_8x8.reshape(-1).copy(), dtype=torch.long)

        return squares_tensor, labels_tensor


# -----------------------------
# Transforms + Dataloaders
# -----------------------------
def get_default_transforms(is_training: bool = True) -> Callable:
    """
    Transforms for square images (already resized to square_size in extract_squares_with_padding).
    Note: We DON'T do RandomRotation here, because we already handle 90/180/270 at the board level.
    """
    from torchvision import transforms  # type: ignore

    if is_training:
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            transforms.RandomAffine(degrees=3, translate=(0.03, 0.03)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            # occlusion-like augmentation (optional but useful):
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int = 8,       # now it's "boards per batch"
    num_workers: int = 4,
    square_size: int = 224,
    board_size: int = 512,
    use_perspective_transform: bool = True
) -> Tuple[DataLoader, DataLoader]:
    train_transform = get_default_transforms(is_training=True)
    val_transform = get_default_transforms(is_training=False)

    train_dataset = ChessboardDataset(
        df=train_df,
        transform=train_transform,
        board_size=board_size,
        square_size=square_size,
        rotation_augment=True,
        use_perspective_transform=use_perspective_transform
    )

    val_dataset = ChessboardDataset(
        df=val_df,
        transform=val_transform,
        board_size=board_size,
        square_size=square_size,
        rotation_augment=False,  # keep validation stable
        use_perspective_transform=use_perspective_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
