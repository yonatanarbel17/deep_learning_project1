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
from tqdm import tqdm


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
        weight_decay: float = 1e-3,
        patience: int = 5,
        lr_scheduler_patience: int = 3,
        class_weights: torch.Tensor = None,
        label_smoothing: float = 0.1,
        unfreeze_epoch: int = 0,
        unfreeze_lr_factor: float = 0.5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        self.patience = patience
        self.lr_scheduler_patience = lr_scheduler_patience
        self.unfreeze_epoch = unfreeze_epoch
        self.unfreeze_lr_factor = unfreeze_lr_factor
        self.unfrozen = False

        os.makedirs(output_dir, exist_ok=True)

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=label_smoothing
            )
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=lr_scheduler_patience,
            factor=0.5
        )

        self.history: List[Dict] = []
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """Run one training epoch. Returns (loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for squares, labels in pbar:
            # squares: (B, 64, 3, H, W), labels: (B, 64)
            squares = squares.to(self.device)
            labels = labels.to(self.device)

            # Reshape for loss computation
            B = squares.shape[0]
            logits = self.model(squares)  # (B, 64, num_classes)

            # Flatten for CrossEntropyLoss: (B*64, 13) vs (B*64,)
            logits_flat = logits.view(-1, logits.shape[-1])
            labels_flat = labels.view(-1)

            loss = self.criterion(logits_flat, labels_flat)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * B
            preds = logits_flat.argmax(dim=1)
            correct += (preds == labels_flat).sum().item()
            total += labels_flat.numel()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")
        
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
        Full training loop with validation, early stopping, and LR scheduling.

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

            # Gradual unfreezing: unfreeze all layers at the specified epoch
            if self.unfreeze_epoch > 0 and epoch == self.unfreeze_epoch and not self.unfrozen:
                for param in self.model.parameters():
                    param.requires_grad = True
                # Rebuild optimizer with all params and reduced LR
                current_lr = self.optimizer.param_groups[0]['lr']
                new_lr = current_lr * self.unfreeze_lr_factor
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=new_lr,
                    weight_decay=self.optimizer.param_groups[0]['weight_decay']
                )
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='max',
                    patience=self.lr_scheduler_patience, factor=0.5
                )
                self.unfrozen = True
                n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"  >>> Unfroze all layers at epoch {epoch} | New LR: {new_lr:.2e} | Trainable params: {n_trainable:,}")

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Step scheduler based on validation accuracy
            self.scheduler.step(val_acc)

            epoch_time = time.time() - epoch_start

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Record history
            epoch_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_sec": epoch_time,
                "lr": current_lr
            }
            self.history.append(epoch_data)

            # Save best model and track improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                torch.save(self.model.state_dict(),
                          os.path.join(self.output_dir, "best_model.pth"))
            else:
                self.epochs_without_improvement += 1

            # Print progress
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping triggered after {self.epochs_without_improvement} epochs without improvement")
                break

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

