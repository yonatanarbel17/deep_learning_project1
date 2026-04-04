"""
Inference module for chess board prediction.
Handles OOD detection via confidence thresholding, supervised occlusion
class (ID 13), and FEN reconstruction.

Hybrid decision logic:
  - If predicted class == 13 (supervised occlusion) -> 'unknown'
  - If confidence < threshold (OOD fallback)         -> 'unknown'
  - Otherwise                                        -> predicted class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, Optional
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.data_loader import grid_to_fen, ID_TO_PIECE, NUM_CLASSES

OCCLUDED_CLASS_ID = 13

# Chess piece count constraints (max legal counts per side)
# ID mapping: 0=P, 1=N, 2=B, 3=R, 4=Q, 5=K, 6=p, 7=n, 8=b, 9=r, 10=q, 11=k
MAX_PIECE_COUNTS = {
    0: 8, 1: 10, 2: 10, 3: 10, 4: 9, 5: 1,   # White
    6: 8, 7: 10, 8: 10, 9: 10, 10: 9, 11: 1,  # Black
}


def apply_chess_constraints(grid: np.ndarray, probs_grid: np.ndarray) -> np.ndarray:
    """
    Apply chess rule constraints to post-process predictions.

    Enforces:
    1. No pawns on ranks 1 or 8
    2. Exactly one king per side (fix using second-best prediction if extra)
    3. Piece counts within legal limits

    Args:
        grid: 8x8 numpy array of predicted class IDs (int or 'unknown')
        probs_grid: (8, 8, num_classes) probability array for fallback corrections

    Returns:
        Corrected 8x8 grid
    """
    grid = grid.copy()

    # --- Rule 1: No pawns on ranks 1 or 8 (rows 0 and 7 in grid) ---
    for col in range(8):
        for row in [0, 7]:
            cell = grid[row, col]
            if isinstance(cell, (int, np.integer)) and cell in (0, 6):  # White pawn=0, Black pawn=6
                # Replace with second-best non-pawn prediction
                probs = probs_grid[row, col].copy()
                probs[0] = 0   # zero out white pawn
                probs[6] = 0   # zero out black pawn
                grid[row, col] = int(np.argmax(probs))

    # --- Rule 2: Exactly one king per side ---
    for king_id in [5, 11]:  # White king=5, Black king=11
        positions = []
        for r in range(8):
            for c in range(8):
                if isinstance(grid[r, c], (int, np.integer)) and int(grid[r, c]) == king_id:
                    positions.append((r, c))

        if len(positions) > 1:
            # Keep the king with highest confidence, fix the rest
            best_pos = max(positions, key=lambda p: probs_grid[p[0], p[1], king_id])
            for r, c in positions:
                if (r, c) != best_pos:
                    probs = probs_grid[r, c].copy()
                    probs[king_id] = 0  # zero out king class
                    grid[r, c] = int(np.argmax(probs))

    # --- Rule 3: Piece count limits ---
    for piece_id, max_count in MAX_PIECE_COUNTS.items():
        positions = []
        for r in range(8):
            for c in range(8):
                if isinstance(grid[r, c], (int, np.integer)) and int(grid[r, c]) == piece_id:
                    positions.append((r, c, probs_grid[r, c, piece_id]))

        if len(positions) > max_count:
            # Sort by confidence (ascending), fix least confident extras
            positions.sort(key=lambda x: x[2])
            for r, c, _ in positions[:len(positions) - max_count]:
                probs = probs_grid[r, c].copy()
                probs[piece_id] = 0
                grid[r, c] = int(np.argmax(probs))

    return grid


class BoardPredictor:
    """
    Predicts board state from square images with hybrid OOD + occlusion handling.
    
    A square is marked 'unknown' when either:
      1. The model confidently predicts the supervised occlusion class (ID 13), or
      2. The model's max confidence falls below the OOD threshold.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        threshold: float = 0.5,
        temperature: float = 1.0
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.threshold = threshold
        self.temperature = temperature

    def calibrate_temperature(self, val_loader: DataLoader, lr: float = 0.01, max_iter: int = 50):
        """
        Learn optimal temperature for confidence calibration using validation set.
        Minimizes NLL on validation data to produce well-calibrated probabilities.
        """
        self.model.eval()
        temperature = nn.Parameter(torch.ones(1, device=self.device) * 1.5)
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
        nll_criterion = nn.CrossEntropyLoss()

        # Collect all logits and labels
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for squares, labels in val_loader:
                squares = squares.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(squares)
                all_logits.append(logits.view(-1, logits.shape[-1]))
                all_labels.append(labels.view(-1))

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        def eval_fn():
            optimizer.zero_grad()
            loss = nll_criterion(all_logits / temperature, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        self.temperature = temperature.item()
        print(f"Calibrated temperature: {self.temperature:.4f}")
    
    @staticmethod
    def _apply_hybrid_filter(predictions: np.ndarray, confidences: np.ndarray, threshold: float) -> np.ndarray:
        """Apply the hybrid occlusion + OOD filter to raw predictions."""
        predictions = predictions.astype(object)
        # Supervised occlusion: model predicts class 13
        predictions[predictions == OCCLUDED_CLASS_ID] = 'unknown'
        # OOD fallback: low confidence on any class
        predictions[confidences < threshold] = 'unknown'
        return predictions
    
    @torch.no_grad()
    def predict_board(
        self,
        squares_tensor: torch.Tensor,
        return_confidences: bool = False,
        apply_constraints: bool = True
    ) -> Tuple[np.ndarray, str, Optional[np.ndarray]]:
        """
        Predict board state from 64 square images.

        Args:
            squares_tensor: (64, 3, H, W) tensor of square images
            return_confidences: Whether to return confidence scores
            apply_constraints: Whether to apply chess rule post-processing

        Returns:
            grid: 8x8 numpy array with class IDs (0-12) or 'unknown'
            fen: FEN string (treats 'unknown' as empty)
            confidences: Optional 8x8 array of confidence scores
        """
        if squares_tensor.dim() == 4:
            squares_tensor = squares_tensor.unsqueeze(0)

        squares_tensor = squares_tensor.to(self.device)

        logits = self.model(squares_tensor)

        # Apply temperature scaling if available
        if self.temperature != 1.0:
            logits = logits / self.temperature

        probs = F.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)

        confidences = confidences.squeeze(0).cpu().numpy()
        predictions = predictions.squeeze(0).cpu().numpy()
        probs_np = probs.squeeze(0).cpu().numpy()  # (64, num_classes)

        predictions = self._apply_hybrid_filter(predictions, confidences, self.threshold)

        grid = predictions.reshape(8, 8)
        confidences_grid = confidences.reshape(8, 8)
        probs_grid = probs_np.reshape(8, 8, -1)

        # Apply chess rule constraints
        if apply_constraints:
            grid = apply_chess_constraints(grid, probs_grid)

        fen = grid_to_fen(grid)

        if return_confidences:
            return grid, fen, confidences_grid
        return grid, fen, None
    
    @torch.no_grad()
    def predict_board_tta(
        self,
        squares_tensor: torch.Tensor,
        apply_constraints: bool = True,
        num_augments: int = 5
    ) -> Tuple[np.ndarray, str, Optional[np.ndarray]]:
        """
        Test-Time Augmentation: run multiple slightly augmented versions of each
        square through the model and average the probabilities before taking argmax.

        Augmentations: original + horizontal flip + small rotations (±3°, ±5°).
        No retraining needed — just smarter inference.

        Args:
            squares_tensor: (64, 3, H, W) tensor of square images
            apply_constraints: Whether to apply chess rule post-processing
            num_augments: Number of augmented versions (default 5)

        Returns:
            grid, fen, confidences (same as predict_board)
        """
        import torchvision.transforms.functional as TF

        if squares_tensor.dim() == 4:
            squares_tensor = squares_tensor.unsqueeze(0)

        squares_tensor = squares_tensor.to(self.device)
        # squares_tensor: (1, 64, 3, H, W)
        sq = squares_tensor.squeeze(0)  # (64, 3, H, W)

        # Build augmented versions
        augmented = [sq]  # original

        # Horizontal flip
        augmented.append(torch.flip(sq, dims=[-1]))

        # Small rotations
        for angle in [3.0, -3.0, 5.0]:
            rotated = torch.stack([TF.rotate(sq[i], angle) for i in range(64)])
            augmented.append(rotated)
            if len(augmented) >= num_augments:
                break

        # Run all through the model, average probabilities
        all_probs = []
        for aug_sq in augmented:
            aug_batch = aug_sq.unsqueeze(0)  # (1, 64, 3, H, W)
            logits = self.model(aug_batch)
            if self.temperature != 1.0:
                logits = logits / self.temperature
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.squeeze(0))  # (64, num_classes)

        # Average probabilities across augmentations
        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)  # (64, num_classes)

        confidences, predictions = torch.max(avg_probs, dim=-1)
        confidences = confidences.cpu().numpy()
        predictions = predictions.cpu().numpy()
        probs_np = avg_probs.cpu().numpy()

        predictions = self._apply_hybrid_filter(predictions, confidences, self.threshold)

        grid = predictions.reshape(8, 8)
        confidences_grid = confidences.reshape(8, 8)
        probs_grid = probs_np.reshape(8, 8, -1)

        if apply_constraints:
            grid = apply_chess_constraints(grid, probs_grid)

        fen = grid_to_fen(grid)
        return grid, fen, confidences_grid

    @torch.no_grad()
    def predict_batch(
        self,
        squares_batch: torch.Tensor
    ) -> Tuple[np.ndarray, list]:
        """
        Predict multiple boards at once.
        
        Args:
            squares_batch: (B, 64, 3, H, W) tensor
            
        Returns:
            grids: (B, 8, 8) array of predictions
            fens: List of B FEN strings
        """
        squares_batch = squares_batch.to(self.device)
        B = squares_batch.shape[0]
        
        logits = self.model(squares_batch)
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)
        
        confidences = confidences.cpu().numpy()
        predictions = predictions.cpu().numpy()
        
        predictions = self._apply_hybrid_filter(predictions, confidences, self.threshold)
        
        grids = predictions.reshape(B, 8, 8)
        fens = [grid_to_fen(grids[i]) for i in range(B)]
        
        return grids, fens
    
    def set_threshold(self, threshold: float):
        """Update the confidence threshold."""
        self.threshold = threshold


def find_optimal_threshold(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    thresholds: Optional[list] = None
) -> Tuple[float, dict]:
    """
    Find the optimal confidence threshold on validation set.
    
    Tests different thresholds and finds the one that maximizes accuracy
    while appropriately flagging low-confidence predictions.
    
    Args:
        model: Trained model
        val_loader: Validation DataLoader
        device: Device to use
        thresholds: List of thresholds to test (default: 0.1 to 0.9)
        
    Returns:
        best_threshold: Optimal threshold value
        results: Dict mapping threshold -> accuracy
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    model.eval()
    model.to(device)
    
    # Collect all predictions and labels
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for squares, labels in val_loader:
            squares = squares.to(device)
            labels = labels.to(device)
            
            logits = model(squares)  # (B, 64, num_classes)
            probs = F.softmax(logits, dim=-1)
            
            all_probs.append(probs.view(-1, probs.shape[-1]))  # (B*64, 13)
            all_labels.append(labels.view(-1))                  # (B*64,)
    
    all_probs = torch.cat(all_probs, dim=0)    # (N, 13)
    all_labels = torch.cat(all_labels, dim=0)  # (N,)
    
    confidences, predictions = torch.max(all_probs, dim=1)
    
    # Test each threshold
    results = {}
    best_threshold = 0.5
    best_accuracy = 0.0
    
    print("\nThreshold Optimization:")
    print("-" * 40)
    
    for t in thresholds:
        # Accuracy: correct predictions among confident ones
        confident_mask = confidences >= t
        if confident_mask.sum() == 0:
            acc = 0.0
            coverage = 0.0
        else:
            correct = (predictions[confident_mask] == all_labels[confident_mask]).float()
            acc = correct.mean().item()
            coverage = confident_mask.float().mean().item()
        
        results[t] = {
            "accuracy": acc,
            "coverage": coverage,
            "score": acc * coverage  # Balance accuracy and coverage
        }
        
        print(f"Threshold {t:.1f}: Acc={acc:.4f}, Coverage={coverage:.4f}, Score={acc*coverage:.4f}")
        
        # Use score (accuracy * coverage) to find best threshold
        if results[t]["score"] > best_accuracy:
            best_accuracy = results[t]["score"]
            best_threshold = t
    
    print("-" * 40)
    print(f"Optimal threshold: {best_threshold} (score={best_accuracy:.4f})")
    
    return best_threshold, results


def get_prediction_summary(grid: np.ndarray) -> dict:
    """
    Get a summary of the predicted board state.
    
    Args:
        grid: 8x8 prediction grid
        
    Returns:
        Dictionary with piece counts, unknown/occluded counts
    """
    summary = {
        "white_pieces": 0,
        "black_pieces": 0,
        "empty": 0,
        "unknown": 0,
        "occluded": 0,
        "pieces": {}
    }
    
    for row in grid:
        for cell in row:
            if cell == 'unknown' or cell == 'occluded':
                summary["unknown"] += 1
            elif cell == OCCLUDED_CLASS_ID:
                summary["occluded"] += 1
            elif cell == 12:
                summary["empty"] += 1
            elif isinstance(cell, (int, np.integer)) and cell <= 5:
                summary["white_pieces"] += 1
                piece = ID_TO_PIECE[int(cell)]
                summary["pieces"][piece] = summary["pieces"].get(piece, 0) + 1
            elif isinstance(cell, (int, np.integer)):
                summary["black_pieces"] += 1
                piece = ID_TO_PIECE[int(cell)]
                summary["pieces"][piece] = summary["pieces"].get(piece, 0) + 1
    
    return summary

