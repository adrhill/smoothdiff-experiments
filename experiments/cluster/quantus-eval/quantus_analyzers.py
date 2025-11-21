import gc
import numpy as np
import torch
import torchvision
import quantus

# Load Captum attributors
from captum.attr import *

# Load SmoothDiff source
from smoothdiff import *

# Use Zennit for LRP
from zennit.composites import EpsilonPlus, EpsilonAlpha2Beta1
from zennit.attribution import Gradient as ZennitGradient

from abc import ABC, abstractmethod

import os
import h5py
import tqdm

N_SAMPLES_DEFAULT = 20

# ==============#
# Base classes #
# ==============#


class AbstractExplainer(ABC):
    def __init__(self, device=None, reduce=True):
        self.reduce = reduce
        super().__init__()

    def __call__(
        self, model, inputs, targets, abs=False, normalise=False, device=None
    ) -> np.ndarray:
        ## Adapted from Quantus tutorial notebook

        # Proactively manage memory
        gc.collect()
        torch.cuda.empty_cache()

        # Prepare inputs and targets
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.Tensor(inputs).reshape(-1, 3, 224, 224).to(device)
        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets).long().to(device)
        assert len(np.shape(inputs)) == 4, (
            "Inputs should be shaped (nr_samples, nr_channels, img_size, img_size) e.g., (1, 3, 224, 224)."
        )
        inputs.requires_grad_(True)

        # Set model in evaluate mode.
        model.to(device)
        model.eval()

        # Run XAI method
        explanation = (
            self.compute_explanation(model, inputs, targets, device)
            .cpu()
            .detach()
            .numpy()
        )

        # Reduce color channel dimension
        if self.reduce:
            explanation = self.reduce_fn(explanation).reshape(-1, 224, 224)

        # Normalize values in explanation
        if normalise:
            explanation = self.normalize_fn(explanation)

        # Convert to absolute values
        if abs:
            explanation = np.abs(explanation)

        # Proactively manage memory
        gc.collect()
        torch.cuda.empty_cache()

        return explanation

    def __str__(self):
        return self.get_name()

    @abstractmethod
    def compute_explanation(self, model, inputs, targets, device):
        pass

    @abstractmethod
    def reduce_fn(self, explanation) -> np.ndarray:
        pass

    @abstractmethod
    def normalize_fn(self, explanation) -> np.ndarray:
        pass

    @abstractmethod
    def heatmap(self, axis, explanation):
        pass

    @abstractmethod
    def get_name(self):
        pass


class SensitivityExplainer(AbstractExplainer):
    def reduce_fn(self, explanation):
        assert explanation.shape[1:] == (3, 224, 224), (
            f"Shape mismatch: expected (3, 224, 224), got {explanation.shape[1:]}"
        )
        return np.linalg.norm(explanation, axis=1)

    def normalize_fn(self, explanation):
        return quantus.normalise_func.normalise_by_max(explanation)

    def heatmap(self, ax, explanation):
        assert explanation.shape == (224, 224), (
            f"Shape mismatch: expected (224, 224), got {explanation.shape}"
        )
        ax.imshow(
            self.normalize_fn(np.abs(explanation)), cmap=cmc.batlow, vmin=0.0, vmax=1.0
        )
        return ax


class AttributionExplainer(AbstractExplainer):
    def reduce_fn(self, explanation):
        assert explanation.shape[1:] == (3, 224, 224), (
            f"Shape mismatch: expected (3, 224, 224), got {explanation.shape[1:]}"
        )
        return np.sum(explanation, axis=1)

    def normalize_fn(self, explanation):
        return quantus.normalise_func.normalise_by_negative(explanation)

    def heatmap(self, ax, explanation):
        assert explanation.shape == (224, 224), (
            f"Shape mismatch: expected (224, 224), got {explanation.shape}"
        )
        ax.imshow(self.normalize_fn(explanation), cmap=cmc.vik, vmin=-1.0, vmax=1.0)
        return ax


# ==============#
# Sensitivity  #
# ==============#


class GradientExplainer(SensitivityExplainer):
    """Wrapper around captum's Saliency implementation."""

    def compute_explanation(self, model, inputs, targets, device):
        return Saliency(model).attribute(inputs, target=targets, abs=False)

    def get_name(self):
        return "Gradient"


class SmoothGradExplainer(SensitivityExplainer):
    def __init__(self, n=N_SAMPLES_DEFAULT, std=0.5, *args, **kwargs):
        self.n = n
        self.std = std
        super().__init__(*args, **kwargs)

    def compute_explanation(self, model, inputs, targets, device):
        gradient_explainer = Saliency(model)

        attribution = torch.zeros_like(inputs)
        for _i in range(self.n):
            inputs_noisy = inputs + torch.randn_like(inputs) * self.std
            attribution += gradient_explainer.attribute(
                inputs_noisy, target=targets, abs=False
            )
        return attribution / self.n

    def __str__(self):
        return f"SmoothGrad (n={self.n}, std={self.std})"

    def get_name(self):
        return "SmoothGrad"


class SmoothDiffExplainer(SensitivityExplainer):
    def __init__(self, n=N_SAMPLES_DEFAULT, std=0.5, *args, **kwargs):
        self.n = n
        self.std = std
        super().__init__(*args, **kwargs)

    def compute_explanation(self, model, inputs, targets, device):
        # Prepare model and set it to evaluate mode.
        # Note: assumes flattened model
        model_smooth = replace_nonlinear_layers(model)
        model_smooth.to(device).eval()

        # collect stats over n noised samples
        set_smoothdiff_layer_mode(
            model_smooth, collect_stats=True, smooth_backward=False
        )
        with torch.no_grad():
            for _i in range(self.n):
                inputs_noisy = inputs + torch.randn_like(inputs) * self.std
                model_smooth(inputs_noisy)

        # calculate smoothed gradients for original image
        set_smoothdiff_layer_mode(
            model_smooth, collect_stats=False, smooth_backward=True
        )
        return Saliency(model_smooth).attribute(inputs, target=targets, abs=False)

    def get_name(self):
        return "SmoothDiff"

    def __str__(self):
        return f"SmoothDiff (n={self.n}, std={self.std})"


class GradCAMExplainer(SensitivityExplainer):
    def __init__(self, layer_index=None, *args, **kwargs):
        self.layer_index = layer_index
        super().__init__(*args, **kwargs)

    def compute_explanation(self, model, inputs, targets, device):
        layer = model[self.layer_index]  # last conv layer in VGG19
        assert isinstance(layer, torch.nn.Conv2d)
        return GuidedGradCam(model, layer).attribute(
            inputs,
            target=targets,
        )

    def get_name(self):
        return "GradCAM"


class IntegratedGradientsExplainer(SensitivityExplainer):
    def __init__(self, n=N_SAMPLES_DEFAULT, *args, **kwargs):
        self.n = n
        super().__init__(*args, **kwargs)

    def compute_explanation(self, model, inputs, targets, device):
        return IntegratedGradients(model).attribute(
            inputs=inputs,
            target=targets,
            baselines=torch.zeros_like(inputs),
            n_steps=self.n,
            internal_batch_size=64,
            method="riemann_trapezoid",
        )

    def __str__(self):
        return f"Integrated Gradients (n={self.n})"

    def get_name(self):
        return "Integrated Gradients"


class RandomNormExplainer(SensitivityExplainer):
    def compute_explanation(self, model, inputs, targets, device):
        return torch.randn(*inputs.shape, device=device)

    def get_name(self):
        return "Random (norm)"


# ==============#
# Attributions #
# ==============#


class InputXGradientExplainer(AttributionExplainer):
    def compute_explanation(self, model, inputs, targets, device):
        return InputXGradient(model).attribute(inputs, target=targets)

    def get_name(self):
        return "Input x Gradient"


class GradientShapExplainer(AttributionExplainer):
    def __init__(self, n=N_SAMPLES_DEFAULT, *args, **kwargs):
        self.n = n
        super().__init__(*args, **kwargs)

    def compute_explanation(self, model, inputs, targets, device):
        baselines = torch.zeros_like(inputs).to(device)
        return GradientShap(model).attribute(
            inputs=inputs, target=targets, n_samples=self.n, baselines=baselines
        )

    def __str__(self):
        return f"GradientShap (n={self.n})"

    def get_name(self):
        return "GradientShap"


class SmoothGradSquareExplainer(AttributionExplainer):
    def __init__(self, n=N_SAMPLES_DEFAULT, std=0.5, *args, **kwargs):
        self.n = n
        self.std = std
        super().__init__(*args, **kwargs)

    def compute_explanation(self, model, inputs, targets, device):
        gradient_explainer = Saliency(model)

        # Implemented according to Hooker et al., "A Benchmark for Interpretability Methods in Deep Neural Networks", p. 6
        attribution = torch.zeros_like(inputs)
        for _i in range(self.n):
            inputs_noisy = inputs + torch.randn_like(inputs) * self.std
            attribution += torch.square(
                gradient_explainer.attribute(inputs_noisy, target=targets, abs=False)
            )
        return attribution / self.n

    def __str__(self):
        return f"SmoothGradSquare (n={self.n}, std={self.std})"

    def get_name(self):
        return "SmoothGradSquare"


class SmoothDiffSquareExplainer(AttributionExplainer):
    def __init__(self, n=N_SAMPLES_DEFAULT, std=0.5, *args, **kwargs):
        self.smoothdiff = SmoothDiffExplainer(n=n, std=std, *args, **kwargs)
        super().__init__(*args, **kwargs)

    def compute_explanation(self, model, inputs, targets, device):
        attribution = self.smoothdiff.compute_explanation(
            model, inputs, targets, device
        )
        return torch.square(attribution)

    def get_name(self):
        return "SmoothDiffSquare"

    def __str__(self):
        return f"SmoothDiffSquare (n={self.smoothdiff.n}, std={self.smoothdiff.std})"


class RandomSumExplainer(AttributionExplainer):
    def compute_explanation(self, model, inputs, targets, device):
        return torch.randn(*inputs.shape, device=device)

    def get_name(self):
        return "Random (sum)"


class RandomSquareExplainer(AttributionExplainer):
    def compute_explanation(self, model, inputs, targets, device):
        return torch.square(torch.randn(*inputs.shape, device=device))

    def get_name(self):
        return "Random (square)"


class LRPCompositeExplainer(AttributionExplainer):
    def __init__(self, composite, *args, **kwargs):
        self.composite = composite
        super().__init__(*args, **kwargs)

    def compute_explanation(self, model, inputs, targets, device):
        target_mat = torch.eye(1000, device=device)[targets]
        with ZennitGradient(model=model, composite=self.composite) as attributor:
            _output, attribution = attributor(inputs, target_mat)

        return attribution

    def get_name(self):
        return f"LRP {type(self.composite).__name__}"


if __name__ == "__main__":
    device = "cpu"

    # Load test data and make loaders.
    f = h5py.File("data/ImageNetS50-128.h5", "r")
    x_batch_np = f["inputs"][:]
    y_batch_np = f["labels"][:]
    s_batch_np = f["masks"][:]

    x_batch, s_batch, y_batch = (
        torch.from_numpy(x_batch_np).to(device),
        torch.from_numpy(s_batch_np).to(device),
        torch.from_numpy(y_batch_np).to(device),
    )

    print(f"{len(x_batch)} matches found.")

    # Load pre-trained and flattened VGG19 model.
    def load_vgg19_flat(device):
        m = torchvision.models.vgg19(weights="IMAGENET1K_V1")
        m = torch.nn.Sequential(
            *m.features, m.avgpool, torch.nn.Flatten(), *m.classifier
        )
        return m.to(device).eval()

    # Load pre-trained ResNet model.
    def load_resnet18(device):
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        return m.to(device).eval()

    model = load_vgg19_flat(device)

    # Load methods
    n_samples = 20
    methods = (
        RandomSumExplainer(),
        GradientExplainer(),
        SmoothDiffExplainer(n=n_samples),
        SmoothGradExplainer(n=n_samples),
        InputXGradientExplainer(),
        GradientShapExplainer(n=n_samples),
        GradCAMExplainer(layer_index=34),  # for VGG19
        IntegratedGradientsExplainer(n=n_samples),
        LRPCompositeExplainer(composite=EpsilonPlus()),
        LRPCompositeExplainer(composite=EpsilonAlpha2Beta1()),
    )

    # Compute explanations
    explanations = {
        m.get_name(): (
            m,
            m(
                model=model,
                inputs=x_batch,
                targets=y_batch,
            ),
        )
        for m in tqdm.tqdm(methods)
    }

    ## Show heatmap
    import matplotlib.pyplot as plt
    import cmcrameri.cm as cmc

    batchsize = 128
    fig, axes = plt.subplots(
        nrows=batchsize,
        ncols=1 + len(explanations),
        figsize=(3 * len(explanations), 2.5 * batchsize),
    )

    # Show input images
    for index in range(batchsize):
        axes[index, 0].imshow(
            np.moveaxis(
                quantus.normalise_func.denormalise(
                    x_batch_np[index],
                    mean=np.array([0.485, 0.456, 0.406]),
                    std=np.array([0.229, 0.224, 0.225]),
                ),
                0,
                -1,
            ),
            vmin=0.0,
            vmax=1.0,
        )
        axes[index, 0].title.set_text(f"ImageNet class {y_batch[index].item()}")
        axes[index, 0].axis("off")

    # Show heatmaps for each method
    for i, (name, vals) in enumerate(explanations.items()):
        method, explanation = vals
        axes[0, i + 1].title.set_text(f"{method.get_name()}")

        for index in range(batchsize):
            a = explanation[index].reshape(224, 224)
            method.heatmap(axes[index, i + 1], a)
            axes[index, i + 1].axis("off")

    plt.savefig("quantus_heatmaps_resnet18.pdf", format="pdf", bbox_inches="tight")
