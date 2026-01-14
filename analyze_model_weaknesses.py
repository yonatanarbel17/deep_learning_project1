#!/usr/bin/env python3
"""
Analyze model weaknesses by examining training results and validation predictions.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.dataset import create_dataloaders, get_default_transforms
from data.data_loader import ID_TO_PIECE, PIECE_TO_ID
from models.classifier import create_model
from training.trainer import get_device
from inference.predictor import BoardPredictor

# Import from train.py
sys.path.insert(0, str(Path(__file__).parent))
from train import load_all_games, split_by_game


def analyze_class_performance(model_path, data_root, device, num_samples=500):
    """Analyze per-class accuracy and confusion matrix."""
    print("Loading model...")
    model = create_model(backbone="resnet18", pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load validation data
    print("Loading validation data...")
    df = load_all_games(data_root)
    train_df, val_df = split_by_game(df, val_ratio=0.2)
    
    val_loader, _ = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        batch_size=1,
        num_workers=0,
        square_size=224,
        board_size=512
    )
    
    all_preds = []
    all_labels = []
    all_confidences = []
    
    print(f"Running inference on {min(num_samples, len(val_df)*64)} samples...")
    with torch.no_grad():
        for idx, (squares, labels) in enumerate(val_loader):
            if idx * 64 >= num_samples:
                break
                
            squares = squares.to(device)
            labels = labels.to(device)
            
            logits = model(squares)  # (1, 64, 13)
            probs = torch.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)
            
            all_preds.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_confidences.extend(confidences.cpu().numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    # Per-class accuracy
    print("\n" + "="*60)
    print("PER-CLASS PERFORMANCE")
    print("="*60)
    
    class_names = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k', 'empty']
    class_accuracies = {}
    class_counts = {}
    class_avg_confidence = {}
    
    for class_id in range(13):
        mask = all_labels == class_id
        if mask.sum() == 0:
            continue
            
        class_count = mask.sum()
        correct = (all_preds[mask] == all_labels[mask]).sum()
        accuracy = correct / class_count
        avg_conf = all_confidences[mask].mean()
        
        class_name = class_names[class_id] if class_id < 12 else 'empty'
        class_accuracies[class_name] = accuracy
        class_counts[class_name] = class_count
        class_avg_confidence[class_name] = avg_conf
    
    # Sort by accuracy
    sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1])
    
    print("\nClass Accuracy (worst to best):")
    print("-" * 60)
    for class_name, acc in sorted_classes:
        count = class_counts[class_name]
        conf = class_avg_confidence[class_name]
        print(f"  {class_name:6s}: {acc*100:5.2f}% (n={count:5d}, avg_conf={conf:.3f})")
    
    # Confusion matrix analysis
    print("\n" + "="*60)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*60)
    
    # Build confusion matrix manually
    cm = np.zeros((13, 13), dtype=int)
    for true_label, pred_label in zip(all_labels, all_preds):
        cm[true_label, pred_label] += 1
    
    # Find most confused pairs
    print("\nTop 10 Most Confused Class Pairs:")
    print("-" * 60)
    confusions = []
    for i in range(13):
        for j in range(13):
            if i != j and cm[i, j] > 0:
                true_class = class_names[i] if i < 12 else 'empty'
                pred_class = class_names[j] if j < 12 else 'empty'
                confusions.append((cm[i, j], true_class, pred_class))
    
    confusions.sort(reverse=True)
    for count, true_class, pred_class in confusions[:10]:
        print(f"  {true_class:6s} → {pred_class:6s}: {count:4d} errors")
    
    return {
        'class_accuracies': class_accuracies,
        'class_counts': class_counts,
        'confusion_matrix': cm,
        'overall_accuracy': (all_preds == all_labels).mean()
    }


def analyze_validation_instability(csv_path):
    """Analyze validation accuracy instability."""
    df = pd.read_csv(csv_path)
    
    val_accs = df['val_acc'].values
    train_accs = df['train_acc'].values
    
    # Calculate instability metrics
    val_std = val_accs.std()
    val_range = val_accs.max() - val_accs.min()
    val_trend = val_accs[-5:].mean() - val_accs[:5].mean()
    
    overfit_gap = train_accs - val_accs
    avg_overfit = overfit_gap.mean()
    max_overfit = overfit_gap.max()
    
    print("\n" + "="*60)
    print("VALIDATION STABILITY ANALYSIS")
    print("="*60)
    print(f"Validation Accuracy Std Dev: {val_std:.4f}")
    print(f"Validation Accuracy Range:   {val_range:.4f}")
    print(f"Validation Trend (last 5 - first 5): {val_trend:.4f}")
    print(f"Average Overfitting Gap:     {avg_overfit:.4f}")
    print(f"Maximum Overfitting Gap:      {max_overfit:.4f}")
    
    # Identify problematic epochs
    print("\nEpochs with Validation Accuracy Drops:")
    print("-" * 60)
    for i in range(1, len(val_accs)):
        if val_accs[i] < val_accs[i-1] - 0.01:  # Drop of more than 1%
            print(f"  Epoch {i+1}: {val_accs[i-1]:.4f} → {val_accs[i]:.4f} (drop: {val_accs[i-1]-val_accs[i]:.4f})")


def analyze_threshold_impact(threshold_file):
    """Analyze how threshold affects performance."""
    with open(threshold_file, 'r') as f:
        lines = f.readlines()
    
    thresholds = []
    accuracies = []
    coverages = []
    
    for line in lines[1:]:  # Skip first line
        parts = line.strip().split(':')
        if len(parts) >= 2:
            thresh = float(parts[0].split('=')[1])
            acc = float(parts[1].split(',')[0].split('=')[1])
            cov = float(parts[1].split(',')[1].split('=')[1])
            thresholds.append(thresh)
            accuracies.append(acc)
            coverages.append(cov)
    
    print("\n" + "="*60)
    print("THRESHOLD IMPACT ANALYSIS")
    print("="*60)
    print("\nThreshold vs Accuracy Trade-off:")
    print("-" * 60)
    for t, acc, cov in zip(thresholds, accuracies, coverages):
        print(f"  Threshold {t:.1f}: Acc={acc:.4f}, Coverage={cov:.4f}, Score={acc*cov:.4f}")
    
    # Find optimal balance
    scores = [acc * cov for acc, cov in zip(accuracies, coverages)]
    best_idx = np.argmax(scores)
    print(f"\nBest threshold (by score): {thresholds[best_idx]:.1f}")
    print(f"  Accuracy: {accuracies[best_idx]:.4f}")
    print(f"  Coverage: {coverages[best_idx]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze model weaknesses")
    parser.add_argument("--model", type=str, 
                       default="/Users/rodrigo/.cursor/worktrees/DL_project/ebs/outputs/best_model.pth",
                       help="Path to model")
    parser.add_argument("--data_root", type=str, default="./data",
                       help="Data root directory")
    parser.add_argument("--training_csv", type=str,
                       default="/Users/rodrigo/.cursor/worktrees/DL_project/ebs/outputs/training_summary.csv",
                       help="Training summary CSV")
    parser.add_argument("--threshold_file", type=str,
                       default="/Users/rodrigo/.cursor/worktrees/DL_project/ebs/outputs/optimal_threshold.txt",
                       help="Optimal threshold file")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to analyze")
    
    args = parser.parse_args()
    
    device = get_device()
    
    print("="*60)
    print("MODEL WEAKNESS ANALYSIS")
    print("="*60)
    
    # 1. Training/Validation Analysis
    if os.path.exists(args.training_csv):
        analyze_validation_instability(args.training_csv)
    
    # 2. Threshold Analysis
    if os.path.exists(args.threshold_file):
        analyze_threshold_impact(args.threshold_file)
    
    # 3. Per-class Performance
    if os.path.exists(args.model) and os.path.exists(args.data_root):
        try:
            results = analyze_class_performance(args.model, args.data_root, device, args.num_samples)
            
            print("\n" + "="*60)
            print("SUMMARY OF WEAK SPOTS")
            print("="*60)
            print("\n1. Worst Performing Classes:")
            sorted_classes = sorted(results['class_accuracies'].items(), key=lambda x: x[1])
            for class_name, acc in sorted_classes[:5]:
                print(f"   - {class_name}: {acc*100:.2f}% accuracy")
            
            print("\n2. Overall Performance:")
            print(f"   - Overall Accuracy: {results['overall_accuracy']*100:.2f}%")
            
        except Exception as e:
            print(f"\nError in class analysis: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
