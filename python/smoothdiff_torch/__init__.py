"""SmoothDiff for PyTorch."""

from .smoothdiff import replace_nonlinear_layers, set_smoothdiff_layer_mode

__all__ = ["replace_nonlinear_layers", "set_smoothdiff_layer_mode"]
