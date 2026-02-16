import pytest
import torch.nn as nn
from smoothdiff_torch.smoothdiff import check_supported_layers


def test_unsupported_layer_raises():
    """check_supported_layers raises ValueError on ViT-style layers."""
    vit_block = nn.Sequential(
        nn.LayerNorm(64),
        nn.MultiheadAttention(64, num_heads=4),
        nn.GELU(),
    )
    with pytest.raises(ValueError, match="Unsupported layer type.*LayerNorm"):
        check_supported_layers(vit_block)


def test_supported_layers_pass():
    """check_supported_layers accepts a simple CNN."""
    cnn = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.AdaptiveAvgPool2d(1),
    )
    check_supported_layers(cnn)  # should not raise
