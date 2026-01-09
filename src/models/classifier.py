"""
Chess square classifier using pretrained ResNet.
Takes 64 squares from a board and classifies each into 13 piece classes.
"""

import torch
import torch.nn as nn
from torchvision import models  # type: ignore


class ChessSquareClassifier(nn.Module):
    """
    Classifies individual chess squares into 13 classes:
    0-5: White pieces (P, N, B, R, Q, K)
    6-11: Black pieces (p, n, b, r, q, k)
    12: Empty
    
    Input: (B, 64, 3, H, W) - B boards, each with 64 squares
    Output: (B, 64, 13) - logits for each square
    """
    
    def __init__(
        self,
        num_classes: int = 13,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Load pretrained backbone
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            feature_dim = 512
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            feature_dim = 512
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Replace final FC layer for 13-class classification
        self.backbone.fc = nn.Linear(feature_dim, num_classes)
        
        # Optionally freeze backbone (only train final layer)
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 64, 3, H, W) - batch of boards, each with 64 square images
               OR (N, 3, H, W) - batch of individual squares
        
        Returns:
            logits: (B, 64, 13) or (N, 13) - class logits for each square
        """
        input_shape = x.shape
        
        if len(input_shape) == 5:
            # Board-level input: (B, 64, 3, H, W)
            B, N, C, H, W = input_shape
            x = x.view(B * N, C, H, W)
            logits = self.backbone(x)
            logits = logits.view(B, N, self.num_classes)
        else:
            # Square-level input: (N, 3, H, W)
            logits = self.backbone(x)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted class indices (0-12) for each square."""
        logits = self.forward(x)
        return logits.argmax(dim=-1)


def create_model(
    backbone: str = "resnet18",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    num_classes: int = 13
) -> ChessSquareClassifier:
    """Factory function to create the model."""
    return ChessSquareClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )

