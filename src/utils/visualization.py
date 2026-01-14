"""
Visualization utilities for training analysis and prediction display.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

# Import for chess diagram (optional)
try:
    import chess
    import chess.svg
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False


def plot_training_curves(
    csv_path: str = "outputs/training_summary.csv",
    output_path: Optional[str] = None
) -> None:
    """
    Plot training and validation curves from training summary.
    
    Args:
        csv_path: Path to training_summary.csv
        output_path: Where to save the plot (default: same dir as csv)
    """
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Loss curves
    ax1 = axes[0]
    ax1.plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=3)
    ax1.plot(df['epoch'], df['val_loss'], 'r--', label='Val Loss', linewidth=3)
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=16)
    ax1.set_title('Training & Validation Loss', fontsize=18)
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=14)
    
    # Accuracy curves
    ax2 = axes[1]
    ax2.plot(df['epoch'], df['train_acc'] * 100, 'g-', label='Train Acc', linewidth=3)
    ax2.plot(df['epoch'], df['val_acc'] * 100, 'orange', linestyle='--', label='Val Acc', linewidth=3)
    ax2.set_xlabel('Epoch', fontsize=16)
    ax2.set_ylabel('Accuracy (%)', fontsize=16)
    ax2.set_title('Training & Validation Accuracy', fontsize=18)
    ax2.legend(fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=14)
    
    # Add best validation accuracy annotation
    best_idx = df['val_acc'].idxmax()
    best_epoch = df.loc[best_idx, 'epoch']
    best_acc = df.loc[best_idx, 'val_acc'] * 100
    ax2.annotate(f'Best: {best_acc:.1f}%\n(Epoch {best_epoch})',
                xy=(best_epoch, best_acc),
                xytext=(best_epoch + 1, best_acc - 5),
                fontsize=14,
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_path = os.path.join(os.path.dirname(csv_path), "training_curves.png")
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {output_path}")
    plt.close()


def visualize_prediction(
    grid: np.ndarray,
    fen: str,
    confidences: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    title: str = "Board Prediction"
) -> None:
    """
    Visualize a predicted board state.
    
    Args:
        grid: 8x8 prediction grid (class IDs or 'unknown')
        fen: Predicted FEN string
        confidences: Optional 8x8 confidence scores
        output_path: Where to save the visualization
        title: Plot title
    """
    # Map class IDs to display characters
    id_to_char = {
        0: '♙', 1: '♘', 2: '♗', 3: '♖', 4: '♕', 5: '♔',
        6: '♟', 7: '♞', 8: '♝', 9: '♜', 10: '♛', 11: '♚',
        12: '·', 'unknown': '?'
    }
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Draw board
    for row in range(8):
        for col in range(8):
            # Square color
            color = '#F0D9B5' if (row + col) % 2 == 0 else '#B58863'
            rect = plt.Rectangle((col, 7-row), 1, 1, facecolor=color)
            ax.add_patch(rect)
            
            # Get piece character
            cell = grid[row, col]
            char = id_to_char.get(cell, '?')
            
            # Color for unknown
            text_color = 'red' if cell == 'unknown' else 'black'
            
            # Draw piece
            ax.text(col + 0.5, 7 - row + 0.5, char,
                   fontsize=56, ha='center', va='center',
                   color=text_color, weight='bold')
            
            # Draw confidence if available
            if confidences is not None:
                conf = confidences[row, col]
                conf_color = 'green' if conf >= 0.7 else 'orange' if conf >= 0.4 else 'red'
                ax.text(col + 0.85, 7 - row + 0.15, f'{conf:.2f}',
                       fontsize=14, ha='right', va='bottom',
                       color=conf_color, weight='bold')
    
    # Labels
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    
    # File labels (a-h)
    for i, label in enumerate('abcdefgh'):
        ax.text(i + 0.5, -0.3, label, fontsize=18, ha='center', weight='bold')
    
    # Rank labels (1-8)
    for i in range(8):
        ax.text(-0.3, i + 0.5, str(i + 1), fontsize=18, ha='center', va='center', weight='bold')
    
    ax.axis('off')
    ax.set_title(f'{title}\nFEN: {fen}', fontsize=20, weight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Prediction visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_training_report(
    csv_path: str = "outputs/training_summary.csv",
    output_dir: str = "outputs"
) -> str:
    """
    Generate a text report summarizing the training.
    
    Returns:
        Report as a string
    """
    df = pd.read_csv(csv_path)
    
    best_idx = df['val_acc'].idxmax()
    best_epoch = int(df.loc[best_idx, 'epoch'])
    best_val_acc = df.loc[best_idx, 'val_acc']
    best_val_loss = df.loc[best_idx, 'val_loss']
    
    final_train_acc = df.iloc[-1]['train_acc']
    final_val_acc = df.iloc[-1]['val_acc']
    
    total_time = df['epoch_time_sec'].sum()
    
    report = f"""
================================================================================
                         TRAINING REPORT
================================================================================

SUMMARY
-------
Total Epochs:        {len(df)}
Total Training Time: {total_time/60:.1f} minutes
Best Epoch:          {best_epoch}

BEST MODEL PERFORMANCE (Epoch {best_epoch})
------------------------------------------
Validation Accuracy: {best_val_acc*100:.2f}%
Validation Loss:     {best_val_loss:.4f}

FINAL MODEL PERFORMANCE (Epoch {len(df)})
------------------------------------------
Training Accuracy:   {final_train_acc*100:.2f}%
Validation Accuracy: {final_val_acc*100:.2f}%
Overfit Gap:         {(final_train_acc - final_val_acc)*100:.2f}%

EPOCH-BY-EPOCH SUMMARY
----------------------
"""
    
    for _, row in df.iterrows():
        report += f"Epoch {int(row['epoch']):2d}: Train={row['train_acc']*100:5.2f}%, Val={row['val_acc']*100:5.2f}%\n"
    
    report += "\n================================================================================"
    
    # Save report
    report_path = os.path.join(output_dir, "training_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Training report saved to: {report_path}")
    return report

