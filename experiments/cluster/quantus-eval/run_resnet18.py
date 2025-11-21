import gc
import numpy as np
import pandas as pd
import torch
import torchvision
import quantus
from zennit.torchvision import ResNetCanonizer

import os
import h5py
import tqdm
from pathlib import Path

# Load Explainers
from quantus_analyzers import *

# Enable GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")
BATCHSIZE = 256

# Load test data and make loaders.
f = h5py.File(f"data/ImageNetS50-{BATCHSIZE}.h5", "r")
x_batch_np = f["inputs"][:]
y_batch_np = f["labels"][:]
s_batch_np = f["masks"][:]

x_batch, s_batch, y_batch = (
    torch.from_numpy(x_batch_np).to(device),
    torch.from_numpy(s_batch_np).to(device),
    torch.from_numpy(y_batch_np).to(device),
)

assert x_batch_np.shape == (BATCHSIZE, 3, 224, 224), (
    f"Shape mismatch: expected (BATCHSIZE, 3, 224, 224), got {x_batch.shape}"
)
assert x_batch.shape == (BATCHSIZE, 3, 224, 224), (
    f"Shape mismatch: expected (BATCHSIZE, 3, 224, 224), got {x_batch.shape}"
)


# Load pre-trained ResNet model.
def load_resnet18(device):
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    return m.to(device).eval()


model = load_resnet18(device)

y_pred = torch.argmax(model(x_batch), dim=1)
assert torch.equal(y_pred, y_batch), "Model predictions don't match target labels"

# Load methods
n_samples = 20
methods = (
    RandomNormExplainer(),
    RandomSumExplainer(),
    RandomSquareExplainer(),
    GradientExplainer(),
    SmoothDiffExplainer(n=n_samples),
    SmoothDiffSquareExplainer(n=n_samples),
    SmoothGradExplainer(n=n_samples),
    SmoothGradSquareExplainer(n=n_samples),
    InputXGradientExplainer(),
    # GradientShapExplainer(n=n_samples),
    IntegratedGradientsExplainer(n=n_samples),
    LRPCompositeExplainer(composite=EpsilonPlus(canonizers=[ResNetCanonizer()])),
    LRPCompositeExplainer(composite=EpsilonAlpha2Beta1(canonizers=[ResNetCanonizer()])),
)

# Load metrics
metrics = {
    "AvgSensitivity": quantus.AvgSensitivity(),  # Robustness, lower is better
    "LocalLipschitzEstimate": quantus.LocalLipschitzEstimate(),  # Robustness, lower is better
    "RelevanceRankAccuracy": quantus.RelevanceRankAccuracy(),  # Localisation, higher is better #
    "RelevanceMassAccuracy": quantus.RelevanceMassAccuracy(),  # Localisation, higher is better #
    "Sparseness": quantus.Sparseness(),  # Complexity, higher is better #
    "Complexity": quantus.Complexity(),  # Complexity, lower is better #
    "RandomLogit": quantus.RandomLogit(),  # Randomisation, higher is better #
    "PointingGame": quantus.PointingGame(),  # PointingGame, higher is better #
}


def run_metric(method, metric_name, metric_fn):
    method_name = method.get_name()
    dir_path = Path(f"results/{BATCHSIZE}/resnet18/{metric_name}")
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path.joinpath(f"{method_name}.h5")

    if os.path.isfile(file_path):
        print(f"- Skipping: File already exists:")
        # Print results because why not
        with h5py.File(file_path, "r") as file:
            mean = file["mean"][()]
            print(f"  - Reading {metric_name} of {method_name} method... {mean}")

    else:
        with h5py.File(file_path, "w") as file:
            file.attrs["metric"] = metric_name
            file.attrs["method"] = method_name

            print(
                f"  - Running... ",
                end="",
                flush=True,
            )

            # Get scores and append results.
            scores = metric_fn(
                model=load_resnet18(device),
                x_batch=x_batch_np,
                y_batch=y_batch_np,
                a_batch=None,
                s_batch=s_batch_np[:, np.newaxis, :, :],  # add color channel
                device=device,
                explain_func=method,
            )

            mean = np.mean(scores)
            print(mean)
            file.create_dataset("mean", data=mean, dtype=np.float64)
            file.create_dataset("raw", data=scores)


for metric_name, metric_fn in metrics.items():
    for method in methods:
        method_name = method.get_name()
        print(f"Evaluating {metric_name} on {method_name}:")

        # Empty cache
        gc.collect()
        torch.cuda.empty_cache()

        # Run experiment
        run_metric(method, metric_name, metric_fn)

        # Empty cache
        gc.collect()
        torch.cuda.empty_cache()
