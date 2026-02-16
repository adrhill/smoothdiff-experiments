# smoothdiff-torch

SmoothDiff implementation in PyTorch.

## Installation

```bash
pip install "smoothdiff-torch @ git+https://github.com/adrhill/smoothdiff-experiments.git#subdirectory=python"
```

## Usage

```python
import smoothdiff_torch as smoothdiff

# Prepare model by replacing non-linear layers
model = smoothdiff.prepare_model(model)

# Collect gradient statistics over multiple noisy forward passes
smoothdiff.set_mode(model, collect_stats=True, smooth_backward=False)
with torch.no_grad():
    for _ in range(n_samples):
        model(inputs + torch.randn_like(inputs) * std)

# Compute SmoothDiff explanation via a single backward pass
smoothdiff.set_mode(model, collect_stats=False, smooth_backward=True)
inputs = inputs.clone().detach().requires_grad_(True)
model(inputs).max().backward()
attribution = inputs.grad

# Free accumulated statistics
smoothdiff.reset_stats(model)
```

See [`examples/smoothdiff.ipynb`](examples/smoothdiff.ipynb) for a full example.
