"""SmoothDiff layer implementations and model preparation utilities."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

_SUPPORTED_LAYERS = {
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.MaxPool2d,
    nn.AdaptiveAvgPool2d,
    nn.Linear,
    nn.Dropout,
}


class _SmoothReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, collect_stats, smooth_backward, grad_local_summed, n_samples):
        if collect_stats:
            grad_local_summed += x > 0
            n_samples += 1
        ctx.smooth_backward = smooth_backward
        if smooth_backward:
            ctx.n_samples = n_samples
            ctx.save_for_backward(grad_local_summed)
        else:
            ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        if ctx.smooth_backward:
            assert ctx.n_samples > 0
            (grad_local_summed,) = ctx.saved_tensors
            smooth_grad = grad_local_summed / ctx.n_samples.item()
            return grad_output * smooth_grad, None, None, None, None
        (x,) = ctx.saved_tensors
        return grad_output * (x > 0), None, None, None, None


class _SmoothMaxPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        collect_stats,
        smooth_backward,
        grad_local_unfolded_summed,
        n_samples,
        kernel_size,
        stride,
        padding,
        dilation,
    ):
        assert x.ndim == 4, (
            "Input must be 4D (N, C, H, W) (for torch.nn.functional.unfold)"
        )
        assert type(kernel_size) is int, (
            "kernel_size must be equal across dimensions"
            " (for torch.nn.functional.unfold)"
        )
        assert type(stride) is int, (
            "stride must be equal across dimensions"
            " (for torch.nn.functional.unfold)"
        )
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.smooth_backward = smooth_backward
        ctx.input_shape = x.shape

        # Unfold input into patches
        unfolded = F.unfold(
            x,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        unfolded = unfolded.view(x.size(0), x.size(1), kernel_size**2, -1)

        # Max pooling over patches
        max_vals, _max_indices = unfolded.max(  # type: ignore[call-overload]
            dim=2, keepdims=True
        )

        with torch.no_grad():
            # Compute mask of max values.
            # In the edge case where multiple values in a patch are equal
            # to the max, this can diverge from MaxPool2d.
            grad_local_unfolded = unfolded == max_vals

            if grad_local_unfolded_summed is None:
                grad_local_unfolded_summed = 0
            if collect_stats:
                grad_local_unfolded_summed += grad_local_unfolded
                n_samples += 1

        if smooth_backward:
            # these determine smooth local gradient:
            ctx.n_samples = n_samples
            ctx.save_for_backward(grad_local_unfolded_summed)
        else:
            # the one-hot vectors are the local derivative:
            ctx.save_for_backward(grad_local_unfolded)

        def calc_output_size(input_size, kernel_size, stride, padding, dilation):
            return (
                input_size + 2 * padding - dilation * (kernel_size - 1) - 1
            ) // stride + 1

        output_h = calc_output_size(x.size(2), kernel_size, stride, padding, dilation)
        output_w = calc_output_size(x.size(3), kernel_size, stride, padding, dilation)

        return max_vals.view(x.size(0), x.size(1), output_h, output_w)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        N, C, H, W = ctx.input_shape

        if ctx.smooth_backward:
            assert ctx.n_samples > 0
            (grad_local_unfolded_summed,) = ctx.saved_tensors
            grad_local_unfolded = grad_local_unfolded_summed / ctx.n_samples.item()
        else:
            (grad_local_unfolded,) = ctx.saved_tensors

        grad_input_unfolded = (
            grad_output.flatten(start_dim=2).unsqueeze(2) * grad_local_unfolded
        )
        grad_input_unfolded = grad_input_unfolded.reshape(
            N, C * ctx.kernel_size**2, -1
        )  # (N, C*k², L)
        grad_input = F.fold(
            grad_input_unfolded,
            output_size=(H, W),
            kernel_size=ctx.kernel_size,
            stride=ctx.stride,
            padding=ctx.padding,
            dilation=ctx.dilation,
        )

        return grad_input, None, None, None, None, None, None, None, None


class _SmoothDiffLayer(nn.Module):
    def __init__(self, collect_stats=False, smooth_backward=False):
        super().__init__()
        self.collect_stats = collect_stats
        self.smooth_backward = smooth_backward
        self.register_buffer("grad_local_sum", None)
        self.register_buffer("n_samples", torch.tensor(0))

    def reset_stats(self):
        self.grad_local_summed = None
        self.n_samples.zero_()  # type: ignore[union-attr]

    def forward(self, x):
        raise NotImplementedError("Must be implemented in subclass")


class _SmoothReLU(_SmoothDiffLayer):
    def forward(self, x):
        if self.grad_local_summed is None:
            # gradients are stored in the form of one sample:
            self.grad_local_summed = torch.zeros_like(x)

        return _SmoothReLUFunction.apply(
            x,
            self.collect_stats,
            self.smooth_backward,
            self.grad_local_summed,
            self.n_samples,
        )


class _SmoothMaxPool2d(_SmoothDiffLayer):
    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        if self.grad_local_summed is None:
            # gradients are stored in the form of one *unfolded* sample:
            unfolded = F.unfold(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            unfolded = unfolded.view(x.size(0), x.size(1), self.kernel_size**2, -1)
            self.grad_local_summed = torch.zeros_like(unfolded)
        return _SmoothMaxPool2dFunction.apply(
            x,
            self.collect_stats,
            self.smooth_backward,
            self.grad_local_summed,
            self.n_samples,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )


def set_smoothdiff_layer_mode(model, collect_stats=None, smooth_backward=None):
    """Set the mode of all SmoothDiff layers in a model.

    Args:
        model: PyTorch model containing SmoothDiff layers.
        collect_stats: If True, layers accumulate local gradient statistics
            during forward passes. If False, stop collecting. None to leave
            unchanged.
        smooth_backward: If True, use accumulated statistics for smooth
            gradients during backward passes. None to leave unchanged.
    """
    for module in model.modules():
        if isinstance(module, _SmoothDiffLayer):
            if collect_stats is not None:
                module.collect_stats = collect_stats
                if collect_stats:
                    module.reset_stats()
            if smooth_backward is not None:
                module.smooth_backward = smooth_backward


def _smooth_layer(layer):
    if isinstance(layer, torch.nn.ReLU):
        return _SmoothReLU()
    if isinstance(layer, torch.nn.MaxPool2d):
        return _SmoothMaxPool2d(
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
        )
    return layer


def _check_supported_layers(model: nn.Module):
    """Check whether all layers in a model are supported by SmoothDiff.

    Raises:
        ValueError: If an unsupported layer type is found.
    """

    def _check_module(module, name="", is_root=False):
        has_children = len(list(module.children())) > 0

        # Only check leaf modules (no children) and skip the root
        if not is_root and not has_children:
            module_type = type(module)
            if module_type not in _SUPPORTED_LAYERS:
                msg = (
                    f"Unsupported layer type found:"
                    f" '{module_type.__name__}'"
                    f" at '{name or 'root'}'.\n"
                    f"Please open a feature request at"
                    f" https://github.com/adrhill/smoothdiff-experiments/issues"
                )
                raise ValueError(msg)

        # Recursively check all child modules
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            _check_module(child_module, full_name, is_root=False)

    _check_module(model, is_root=True)


def replace_nonlinear_layers(model):
    """Replace ReLU and MaxPool2d layers with SmoothDiff equivalents.

    Creates a deep copy of the model and recursively replaces all ReLU and
    MaxPool2d layers with their SmoothDiff counterparts.

    Args:
        model: PyTorch model to prepare for SmoothDiff.

    Returns:
        A new model with non-linear layers replaced.

    Raises:
        ValueError: If the model contains unsupported layer types.
    """
    _check_supported_layers(model)
    model_copy = copy.deepcopy(model)

    def _replace_in_module(module):
        for name, child in module.named_children():
            # First, recursively process children
            _replace_in_module(child)

            # Then replace the current child if it matches our criteria
            new_layer = _smooth_layer(child)
            if new_layer is not child:  # Only replace if it actually changed
                setattr(module, name, new_layer)

    _replace_in_module(model_copy)
    return model_copy
