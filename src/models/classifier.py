"""
Chess square classifier using pretrained ResNet or EfficientNet.
Takes 64 squares from a board and classifies each into piece classes.
Optionally uses board-level Transformer context so squares inform each other.
"""

import torch
import torch.nn as nn
from torchvision import models  # type: ignore


class ChessSquareClassifier(nn.Module):
    """
    Classifies individual chess squares into 14 classes:
    0-5: White pieces (P, N, B, R, Q, K)
    6-11: Black pieces (p, n, b, r, q, k)
    12: Empty
    13: Occluded

    Input: (B, 64, 3, H, W) - B boards, each with 64 squares
    Output: (B, 64, 14) - logits for each square

    When use_board_context=True, a Transformer encoder lets each square
    attend to all other squares on the board before classification.
    """

    def __init__(
        self,
        num_classes: int = 14,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
        freeze_ratio: float = 0.0,
        use_board_context: bool = False,
        context_dim: int = 256,
        context_heads: int = 4,
        context_layers: int = 2
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.use_board_context = use_board_context

        # Load pretrained backbone
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            feature_dim = 512
            self._classifier_attr = "fc"
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            feature_dim = 512
            self._classifier_attr = "fc"
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            feature_dim = 2048
            self._classifier_attr = "fc"
        elif backbone == "efficientnet_b2":
            weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b2(weights=weights)
            feature_dim = 1408
            self._classifier_attr = "classifier"
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.feature_dim = feature_dim

        if use_board_context:
            # Remove the backbone's original classifier - we'll use our own
            if self._classifier_attr == "fc":
                self.backbone.fc = nn.Identity()
            else:
                self.backbone.classifier = nn.Identity()

            # Project backbone features to transformer dim
            self.feature_proj = nn.Linear(feature_dim, context_dim)

            # Learnable positional embeddings for 64 squares (8x8 board)
            self.pos_embedding = nn.Parameter(torch.randn(1, 64, context_dim) * 0.02)

            # Transformer encoder for board-level context
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=context_dim,
                nhead=context_heads,
                dim_feedforward=context_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.board_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=context_layers
            )

            # Classification head
            self.classifier_head = nn.Sequential(
                nn.LayerNorm(context_dim),
                nn.Dropout(p=dropout),
                nn.Linear(context_dim, num_classes)
            )
        else:
            # Original per-square classification (no board context)
            if self._classifier_attr == "fc":
                self.backbone.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(feature_dim, num_classes)
                )
            else:
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(feature_dim, num_classes)
                )

        # Optionally freeze backbone (only train final layer)
        if freeze_backbone:
            classifier_attr = self._classifier_attr
            for name, param in self.backbone.named_parameters():
                if classifier_attr not in name:
                    param.requires_grad = False

        # Partial freezing: freeze the first freeze_ratio of backbone layers
        if freeze_ratio > 0.0 and not freeze_backbone:
            self._freeze_early_layers(freeze_ratio)

    def _freeze_early_layers(self, ratio: float):
        """Freeze the first `ratio` fraction of backbone parameters (excluding classifier)."""
        classifier_attr = self._classifier_attr
        all_params = [
            (name, param) for name, param in self.backbone.named_parameters()
            if classifier_attr not in name
        ]
        n_freeze = int(len(all_params) * ratio)
        for i, (name, param) in enumerate(all_params):
            if i < n_freeze:
                param.requires_grad = False
        total = len(all_params)
        print(f"  Froze {n_freeze}/{total} backbone parameter groups ({ratio:.0%})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 64, 3, H, W) - batch of boards, each with 64 square images
               OR (N, 3, H, W) - batch of individual squares

        Returns:
            logits: (B, 64, num_classes) or (N, num_classes) - class logits for each square
        """
        input_shape = x.shape

        if len(input_shape) == 5:
            B, N, C, H, W = input_shape
            x = x.view(B * N, C, H, W)

            if self.use_board_context:
                # Extract features without classifying
                features = self.backbone(x)  # (B*64, feature_dim)
                features = features.view(B, N, self.feature_dim)  # (B, 64, feature_dim)

                # Project to transformer dim and add positional info
                features = self.feature_proj(features)  # (B, 64, context_dim)
                features = features + self.pos_embedding  # (B, 64, context_dim)

                # Board-level attention
                features = self.board_transformer(features)  # (B, 64, context_dim)

                # Classify each square
                logits = self.classifier_head(features)  # (B, 64, num_classes)
            else:
                logits = self.backbone(x)
                logits = logits.view(B, N, self.num_classes)
        else:
            if self.use_board_context:
                # Single squares - no board context available, classify directly
                features = self.backbone(x)  # (N, feature_dim)
                features = self.feature_proj(features).unsqueeze(0)  # (1, N, context_dim)
                features = self.board_transformer(features)
                logits = self.classifier_head(features).squeeze(0)  # (N, num_classes)
            else:
                logits = self.backbone(x)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted class indices for each square."""
        logits = self.forward(x)
        return logits.argmax(dim=-1)


def create_model(
    backbone: str = "resnet18",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    num_classes: int = 14,
    dropout: float = 0.3,
    freeze_ratio: float = 0.0,
    use_board_context: bool = False,
    context_dim: int = 256,
    context_heads: int = 4,
    context_layers: int = 2
) -> ChessSquareClassifier:
    """Factory function to create the model."""
    return ChessSquareClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        freeze_ratio=freeze_ratio,
        use_board_context=use_board_context,
        context_dim=context_dim,
        context_heads=context_heads,
        context_layers=context_layers
    )

