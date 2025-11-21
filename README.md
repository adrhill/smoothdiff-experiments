# SmoothDiff

Code for the NeurIPS 2025 paper *"Smoothed Differentiation Efficiently Mitigates Shattered Gradients in Explanations"*.

## Code

* The Julia reference implementation of SmoothDiff can be found in the [`/julia`](/julia) folder, which contains a Julia package called `SmoothedDifferentiation.jl`.
* The PyTorch reference implementation of SmoothDiff can be found in [`/python`](/python) folder.

## Installation

1. [Install Julia](https://julialang.org/downloads/) `v1.11`. On Unix systems, this requires running
    ```bash
    curl -fsSL https://install.julialang.org | sh
    ```
2. Start a [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/) session by typing `julia` in your terminal
3. Install [DrWatson.jl](https://github.com/JuliaDynamics/DrWatson.jl) by typing the following in your Julia REPL:
    ```julia-repl
    ]add DrWatson
    ```
4. Run the experiments and plotting scripts listed below by typing `include("path/to/file.jl")` in your REPL, replacing the string with the correct path.

## Experiments

We provide all code and virtual environments required to reproduce our experiments and figures.

### Running experiments

The following steps need to be run in sequence:

1. [Download the ImageNet dataset](https://image-net.org/download.php) by agreeing to its terms of access and run the script [`/experiments/heatmaps/save_input.jl`](/experiments/heatmaps/save_input.jl) to save preprocessed input tensors.
2. Compute explanations by running [`/experiments/cluster/run_analyzers/run.jl`](/experiments/cluster/run_analyzers/run.jl) as well as the $n=10^6$ sample SmoothGrad run in [`/experiments/cluster/run_analyzers/run_100k.jl`](/experiments/cluster/run_analyzers/run_100k.jl).
3. Compute benchmarks by running [`/experiments/cluster/run_analyzers/run.jl`](/experiments/cluster/run_analyzers/run_benchmarks.jl).
4. After having computed explanations in step 3, evaluate them using pixel-flipping by running [`/experiments/cluster/pixelflipping/run.jl`](/experiments/cluster/pixelflipping/run.jl).
5. The raw quantitative results from Appendix F can be found in [`/experiments/cluster/quantus-eval/results`](/experiments/cluster/quantus-eval/results) and are computed by running [`/experiments/cluster/quantus-eval/run.sh`](/experiments/cluster/quantus-eval/run.sh).

### Reproducing figures

After having computed explanations and pixel-flipping results, the following plots and tables can be reproduced by running the respective file:

- **Figure 1 (a)**: create data by running [`/experiments/shattered_gradient_1d/run_fig_1a.jl`](/experiments/shattered_gradient_1d/run_fig_1a.jl), then plot using [`/experiments/shattered_gradient_1d/plot_fig_1a.jl`](/experiments/shattered_gradient_1d/plot_fig_1a.jl) 
- **Figure 1 (b)**: run [`/experiments/heatmaps/grid_fig_1b.jl`](/experiments/heatmaps/grid_fig_1b.jl)
- **Figure 2**: run [`/experiments/ssim_convergence/convergence_ssim_combined.jl`](/experiments/ssim_convergence/convergence_ssim_combined.jl)
- **Figure 3** and **Figure 15**: run [`/experiments/heatmaps/grid_convergence.jl`](/experiments/heatmaps/grid_convergence.jl)
- **Figure 4** and **Table 1**: run [`/experiments/pixelflipping_plots/srg_plot.jl`](/experiments/pixelflipping_plots/srg_plot.jl)
- **Figure 5**: run [`/experiments/colormaps/jetvsbatlow.jl`](/experiments/colormaps/jetvsbatlow.jl)
- **Figure 6**: run [`/experiments/heatmaps/grid_pipeline_comparison.jl`](/experiments/heatmaps/grid_pipeline_comparison.jl)
- **Figure 7**: run [`/experiments/ssim_convergence/convergence_cosine_sim.jl`](/experiments/ssim_convergence/convergence_cosine_sim.jl)
- **Figures 8 to 13**: run [`/experiments/cluster/quantus-eval/plot.jl`](/experiments/cluster/quantus-eval/plot.jl)
- **Figure 14**: run [`/experiments/beta-smoothing/plot_beta_smoothing.jl`](/experiments/beta-smoothing/plot_beta_smoothing.jl)
- **Figure 16**: run [`/experiments/heatmaps/grid_multiclass.jl`](/experiments/heatmaps/grid_multiclass.jl)
- **Figure 17**: run [`/experiments/heatmaps/grid_appendix.jl`](/experiments/heatmaps/grid_appendix.jl)
