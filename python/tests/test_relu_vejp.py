"""Tests for the ReLU VEJP."""

import torch

from smoothdiff_torch.smoothdiff import _SmoothReLU


def test_relu_vejp():
    """Test the ReLU VEJP with 5 manual inputs.

    The first 4 inputs are used for stats collection, and the 5th input
    is used for the smooth backward pass. The expected smooth gradient
    is [1, 0.75, 0.5, 0.25, 0].
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

    # Phase 2: Smooth backward pass
    layer.collect_stats = False
    layer.smooth_backward = True
    output = layer(test_input)
    output.sum().backward()

    expected = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    assert torch.allclose(test_input.grad, expected), (  # type: ignore[arg-type]
        f"Expected {expected}, got {test_input.grad}"
    )
