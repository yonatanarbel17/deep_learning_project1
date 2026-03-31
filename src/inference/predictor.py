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
        threshold: float = 0.5
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.threshold = threshold
    
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
        return_confidences: bool = False
    ) -> Tuple[np.ndarray, str, Optional[np.ndarray]]:
        """
        Predict board state from 64 square images.
        
        Args:
            squares_tensor: (64, 3, H, W) tensor of square images
            return_confidences: Whether to return confidence scores
            
        Returns:
            grid: 8x8 numpy array with class IDs (0-12) or 'unknown'
            fen: FEN string (treats 'unknown' as empty)
            confidences: Optional 8x8 array of confidence scores
        """
        if squares_tensor.dim() == 4:
            squares_tensor = squares_tensor.unsqueeze(0)
        
        squares_tensor = squares_tensor.to(self.device)
        
        logits = self.model(squares_tensor)
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)
        
        confidences = confidences.squeeze(0).cpu().numpy()
        predictions = predictions.squeeze(0).cpu().numpy()
        
        predictions = self._apply_hybrid_filter(predictions, confidences, self.threshold)
        
        grid = predictions.reshape(8, 8)
        confidences_grid = confidences.reshape(8, 8)
        fen = grid_to_fen(grid)
        
        if return_confidences:
            return grid, fen, confidences_grid
        return grid, fen, None
    
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

