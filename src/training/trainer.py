"""
Training engine for chess square classifier.
Handles training loop, validation, and saves training summary.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


class Trainer:
    """
    Handles model training with validation and logging.
    
    Produces:
        - training_summary.csv: Epoch-by-epoch metrics
        - best_model.pth: Model with best validation accuracy
        - final_model.pth: Model after all epochs
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: str = "outputs",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.history: List[Dict] = []
        self.best_val_acc = 0.0
    
    def train_epoch(self) -> Tuple[float, float]:
        """Run one training epoch. Returns (loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for squares, labels in self.train_loader:
            # squares: (B, 64, 3, H, W), labels: (B, 64)
            squares = squares.to(self.device)
            labels = labels.to(self.device)
            
            # Reshape for loss computation
            B = squares.shape[0]
            logits = self.model(squares)  # (B, 64, 13)
            
            # Flatten for CrossEntropyLoss: (B*64, 13) vs (B*64,)
            logits_flat = logits.view(-1, logits.shape[-1])
            labels_flat = labels.view(-1)
            
            loss = self.criterion(logits_flat, labels_flat)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * B
            preds = logits_flat.argmax(dim=1)
            correct += (preds == labels_flat).sum().item()
            total += labels_flat.numel()
        
        avg_loss = total_loss / len(self.train_loader.dataset)
        accuracy = correct / total
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Run validation. Returns (loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for squares, labels in self.val_loader:
            squares = squares.to(self.device)
            labels = labels.to(self.device)
            
            B = squares.shape[0]
            logits = self.model(squares)
            
            logits_flat = logits.view(-1, logits.shape[-1])
            labels_flat = labels.view(-1)
            
            loss = self.criterion(logits_flat, labels_flat)
            
            total_loss += loss.item() * B
            preds = logits_flat.argmax(dim=1)
            correct += (preds == labels_flat).sum().item()
            total += labels_flat.numel()
        
        avg_loss = total_loss / len(self.val_loader.dataset)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def train(self, num_epochs: int = 15) -> pd.DataFrame:
        """
        Full training loop with validation.
        
        Returns:
            DataFrame with training history
        """
        print(f"Training on {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)} boards ({len(self.train_loader.dataset)*64} squares)")
        print(f"Val samples: {len(self.val_loader.dataset)} boards ({len(self.val_loader.dataset)*64} squares)")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Record history
            epoch_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_sec": epoch_time
            }
            self.history.append(epoch_data)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), 
                          os.path.join(self.output_dir, "best_model.pth"))
            
            # Print progress
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Time: {epoch_time:.1f}s")
        
        total_time = time.time() - start_time
        print("-" * 60)
        print(f"Training complete in {total_time/60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Save final model
        torch.save(self.model.state_dict(), 
                  os.path.join(self.output_dir, "final_model.pth"))
        
        # Save training summary
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(os.path.join(self.output_dir, "training_summary.csv"), index=False)
        
        return history_df


def get_device() -> torch.device:
    """Get the best available device (MPS for Mac, CUDA for GPU, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

