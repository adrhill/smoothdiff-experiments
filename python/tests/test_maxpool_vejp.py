import torch
from smoothdiff_torch.smoothdiff import SmoothMaxPool2d


def test_maxpool2d_vejp():
    """Test the MaxPool2d VEJP with 4 manual 2x2 inputs.

    Uses kernel_size=2, stride=2 on 1x1x2x2 inputs, producing a scalar output.
    The max is at the top-left position in 3 of 4 stats inputs and at the
    top-right in 1 of 4, so the expected smooth gradient is:
        [[0.75, 0.25],
         [0.00, 0.00]]
    """
    stats_inputs = [
        # max at (0,0)
        torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
        # max at (0,0)
        torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
        # max at (0,0)
        torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
        # max at (0,1)
        torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]]),
    ]
    test_input = torch.tensor(
        [[[[1.0, 1.0], [1.0, 1.0]]]], requires_grad=True
    )

    layer = SmoothMaxPool2d(kernel_size=2, stride=2, padding=0)
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

    expected = torch.tensor([[[[0.75, 0.25], [0.0, 0.0]]]])
    assert torch.allclose(test_input.grad, expected), (
        f"Expected {expected}, got {test_input.grad}"
    )
