"""Tests for the ReLU VEJP."""

import torch

from smoothdiff_torch.smoothdiff import _SmoothReLU


def test_relu_vejp():
    """Test the ReLU VEJP with 5 manual inputs.

    All 5 inputs count toward the statistics (n=5): 4 stats-only inputs
    plus the backward-pass input. The expected smooth gradient is
    count / n = [5, 4, 3, 2, 1] / 5 = [1.0, 0.8, 0.6, 0.4, 0.2].
    """
    stats_inputs = [
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0]),
    ]
    test_input = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], requires_grad=True)

    layer = _SmoothReLU()
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

    expected = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2])
    assert torch.allclose(test_input.grad, expected), (  # type: ignore[arg-type]
        f"Expected {expected}, got {test_input.grad}"
    )
