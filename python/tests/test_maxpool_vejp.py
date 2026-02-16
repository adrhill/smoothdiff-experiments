"""Tests for the MaxPool2d VEJP."""

import torch

from smoothdiff_torch.smoothdiff import _SmoothMaxPool2d


def test_maxpool2d_k2s2_vejp():
    """Test the MaxPool2d VEJP with kernel_size=2, stride=2 on 2x2 inputs.

    10 inputs total (n=10): 9 stats-only inputs + 1 backward-pass input.
    The test input has a clear max at (0,0) to avoid tie-breaking differences.
    Expected counts: [[4, 3], [2, 1]], n=10 → VEJP = [[0.4, 0.3], [0.2, 0.1]]
    """
    stats_inputs = [
        torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),  # 3x max at (0,0)
        torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
        torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
        torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]]),  # 3x max at (0,1)
        torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]]),
        torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]]),
        torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]]),  # 2x max at (1,0)
        torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]]),
        torch.tensor([[[[0.0, 0.0], [0.0, 1.0]]]]),  # 1x max at (1,1)
    ]
    test_input = torch.tensor([[[[4.0, 1.0], [1.0, 1.0]]]], requires_grad=True)

    layer = _SmoothMaxPool2d(kernel_size=2, stride=2, padding=0)
    layer.reset_stats()

    layer.collect_stats = True
    layer.smooth_backward = False
    for x in stats_inputs:
        layer(x)

    layer.collect_stats = True
    layer.smooth_backward = True
    output = layer(test_input)
    output.sum().backward()

    expected = torch.tensor([[[[0.4, 0.3], [0.2, 0.1]]]])
    assert torch.allclose(test_input.grad, expected), (  # type: ignore[arg-type]
        f"Expected {expected}, got {test_input.grad}"
    )


def test_maxpool2d_k3s3_vejp():
    """Test the MaxPool2d VEJP with kernel_size=3, stride=3 on 3x3 inputs.

    10 inputs total (n=10): 9 stats-only inputs + 1 backward-pass input.
    The test input has a clear max at (0,0) to avoid tie-breaking differences.
    Expected counts: [[4, 0, 2], [0, 3, 0], [1, 0, 0]], n=10
    → VEJP = [[0.4, 0.0, 0.2], [0.0, 0.3, 0.0], [0.1, 0.0, 0.0]]
    """
    stats_inputs = [
        torch.tensor(
            [[[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
        ),  # 3x (0,0)
        torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]),
        torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]),
        torch.tensor(
            [[[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
        ),  # 2x (0,2)
        torch.tensor([[[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]),
        torch.tensor(
            [[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]]
        ),  # 3x (1,1)
        torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]]),
        torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]]),
        torch.tensor(
            [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]]
        ),  # 1x (2,0)
    ]
    test_input = torch.tensor(
        [[[[9.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], requires_grad=True
    )

    layer = _SmoothMaxPool2d(kernel_size=3, stride=3, padding=0)
    layer.reset_stats()

    layer.collect_stats = True
    layer.smooth_backward = False
    for x in stats_inputs:
        layer(x)

    layer.collect_stats = True
    layer.smooth_backward = True
    output = layer(test_input)
    output.sum().backward()

    expected = torch.tensor([[[[0.4, 0.0, 0.2], [0.0, 0.3, 0.0], [0.1, 0.0, 0.0]]]])
    assert torch.allclose(test_input.grad, expected), (  # type: ignore[arg-type]
        f"Expected {expected}, got {test_input.grad}"
    )
