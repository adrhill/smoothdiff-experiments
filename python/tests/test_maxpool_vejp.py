"""Tests for the MaxPool2d VEJP."""

import torch

from smoothdiff_torch.smoothdiff import _SmoothMaxPool2d


def test_maxpool2d_vejp():
    """Test the MaxPool2d VEJP with 10 manual 2x2 inputs.

    Uses kernel_size=2, stride=2 on 1x1x2x2 inputs, producing a scalar output.
    All 10 inputs count toward the statistics (n=10): 9 stats-only inputs
    plus the backward-pass input. The test input has a clear max at (0,0)
    to avoid tie-breaking differences. Expected counts:
        [[4, 3], [2, 1]], n=10 → VEJP = [[0.4, 0.3], [0.2, 0.1]]
    """
    stats_inputs = [
        # 3x max at (0,0)
        torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
        torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
        torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
        # 3x max at (0,1)
        torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]]),
        torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]]),
        torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]]),
        # 2x max at (1,0)
        torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]]),
        torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]]),
        # 1x max at (1,1)
        torch.tensor([[[[0.0, 0.0], [0.0, 1.0]]]]),
    ]
    test_input = torch.tensor([[[[4.0, 1.0], [1.0, 1.0]]]], requires_grad=True)

    layer = _SmoothMaxPool2d(kernel_size=2, stride=2, padding=0)
    layer.reset_stats()

    # Phase 1: Collect stats
    layer.collect_stats = True
    layer.smooth_backward = False
    for x in stats_inputs:
        layer(x)

    # Phase 2: Smooth backward pass (still collecting stats so test input counts)
    layer.collect_stats = True
    layer.smooth_backward = True
    output = layer(test_input)
    output.sum().backward()

    expected = torch.tensor([[[[0.4, 0.3], [0.2, 0.1]]]])
    assert torch.allclose(test_input.grad, expected), (  # type: ignore[arg-type]
        f"Expected {expected}, got {test_input.grad}"
    )
