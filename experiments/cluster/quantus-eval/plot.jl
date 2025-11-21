using Pkg
Pkg.activate(@__DIR__)

using DataFrames
using HDF5
using StatsBase
using CairoMakie
using Printf
using Colors

include(joinpath(@__DIR__, "..", "..", "theme.jl"))
set_theme!(theme_smoothdiff())

# Directory structure: `results/$batch_size/$model_name/$metric_name/$method_name`
RESULTS_DIR = joinpath(@__DIR__, "results")
BATCH_SIZE = 256

color_intg = _wong_colors[4]
color_ixg = _wong_colors[5]
color_lrp = _wong_colors[6]
color_gradcam = _wong_colors[7]
color_random = Gray(0.7)

METHOD_NAMES_VGG = reverse(
    [
        ("Gradient", "Gradient (ℓ²)", color_gradient),
        ("Integrated Gradients", "Integrated Gradients (ℓ²)", color_intg),
        ("SmoothDiff", "SmoothDiff (ℓ²)", color_smoothdiff),
        ("SmoothGrad", "SmoothGrad (ℓ²)", color_smoothgrad),
        ("Random (norm)", "Random (ℓ²)", color_random),
        ("SmoothDiffSquare", "SmoothDiff-Squared (Σ)", color_smoothdiff),
        ("SmoothGradSquare", "SmoothGrad-Squared (Σ)", color_smoothgrad),
        ("Random (square)", "Random-Squared (Σ)", color_random),
        ("Input x Gradient", "Input x Gradient (Σ)", color_ixg),
        ("LRP EpsilonPlus", "LRP EpsilonPlus (Σ)", color_lrp),
        ("LRP EpsilonAlpha2Beta1", "LRP EpsilonAlpha2Beta1 (Σ)", color_lrp),
        ("GradCAM", "GradCAM (Σ)", color_gradcam),
        ("Random (sum)", "Random (Σ)", color_random),
    ]
)

METHOD_NAMES_RESNET = reverse(
    [
        ("Gradient", "Gradient (ℓ²)", color_gradient),
        ("Integrated Gradients", "Integrated Gradients (ℓ²)", color_intg),
        ("SmoothDiff", "SmoothDiff (ℓ²)", color_smoothdiff),
        ("SmoothGrad", "SmoothGrad (ℓ²)", color_smoothgrad),
        ("Random (norm)", "Random (ℓ²)", color_random),
        ("SmoothDiffSquare", "SmoothDiff-Squared (Σ)", color_smoothdiff),
        ("SmoothGradSquare", "SmoothGrad-Squared (Σ)", color_smoothgrad),
        ("Random (square)", "Random-Squared (Σ)", color_random),
        ("Input x Gradient", "Input x Gradient (Σ)", color_ixg),
        ("LRP EpsilonPlus", "LRP EpsilonPlus (Σ)", color_lrp),
        ("LRP EpsilonAlpha2Beta1", "LRP EpsilonAlpha2Beta1 (Σ)", color_lrp),
        ("Random (sum)", "Random (Σ)", color_random),
    ]
)

METRIC_NAMES = [
    ("AvgSensitivity", "Average Sensitivity"),
    ("Complexity", "Complexity"),
    ("Sparseness", "Sparseness"),
    ("RelevanceRankAccuracy", "Relevance Rank Accuracy"),
    ("LocalLipschitzEstimate", "Local Lipschitz Estimate"),
    ("GridPG", "Grid Pointing Game"),

]

function load_result(model_name, metric_filename, method_filename)
    path = joinpath(RESULTS_DIR, string(BATCH_SIZE), model_name, metric_filename, method_filename * ".h5")
    return h5open(path, "r") do file
        results = read(file)
        raw = results["raw"]
        if length(raw) != BATCH_SIZE
            error
        end
        mean = results["mean"]
        return (raw = raw, mean = mean, std = StatsBase.std(raw))
    end
end

## Plotting functions
function plot_metric!(
        grid, model_filename, model_name, method_names, metric_filename, metric_name; xscale = identity, xticklabelspace = 3.5mm, show_xlabel = true
    )
    method_ticks = (1:length(method_names), [name for (_filename, name) in method_names])

    ax_box = Axis(grid[1, 1]; xlabel = "$(metric_name) Score", xticksize = 0, xticklabelspace, yticks = method_ticks, yticksize = 0, yticklabelsize = 10, xscale = xscale)
    ax_means = Axis(grid[1, 2]) # align mean values to boxplots
    linkyaxes!(ax_box, ax_means)


    for (i, (method_filename, method_name, color)) in enumerate(method_names)
        res = load_result(model_filename, metric_filename, method_filename)
        categories = fill(i, BATCH_SIZE)
        boxplot!(
            ax_box, categories, res.raw;
            orientation = :horizontal,
            gap = 0.15,
            label = method_name,
            mediancolor = :black,
            color = color,
        )
        text_mean = @sprintf("%.3f", res.mean)
        text!(ax_means, 0, i; text = text_mean, fontsize = 10, align = (:left, :center))
    end
    Label(grid[1, 1, Top()], model_name, valign = :bottom, halign = :center, font = :bold)
    Label(grid[1, 2, Top()], "mean", valign = :bottom, halign = :left, font = :regular, fontsize = 10, padding = (0, 0, -5, 0))

    xlims!(ax_means, 0, 1)
    hidedecorations!(ax_means)
    colsize!(grid, 2, 8mm)
    colgap!(grid, 2mm)
    !show_xlabel && hidexdecorations!(ax_box, label = true, ticklabels = false, ticks = false, grid = false, minorgrid = false, minorticks = false)

    return grid
end

function plot_metric(metric_filename; metric_name = metric_filename, xscale = identity, save_figure = true, display_figure = true)
    fig = Figure(size = (LINE_WIDTH_NEURIPS, 9.5cm), figure_padding = (0, 1mm, 0, 0))
    grid_vgg = fig[1, 1] = GridLayout()
    grid_resnet = fig[2, 1] = GridLayout()
    rowsize!(fig.layout, 1, Auto(12))
    rowsize!(fig.layout, 2, Auto(11))

    plot_metric!(grid_vgg, "vgg19", "VGG-19", METHOD_NAMES_VGG, metric_filename, metric_name; xscale, show_xlabel = false, xticklabelspace = 0.5mm)
    plot_metric!(grid_resnet, "resnet18", "ResNet-18", METHOD_NAMES_RESNET, metric_filename, metric_name; xscale)

    save_figure && save(joinpath(@__DIR__, "figures", "metric_$(metric_filename).svg"), fig)
    display_figure && display(fig)
    return fig
end

## Plot VGG
plot_metric("Complexity")
plot_metric("Sparseness", xscale = identity)
plot_metric("RelevanceRankAccuracy", metric_name = "Relevance Rank Accuracy")
plot_metric("AvgSensitivity", metric_name = "Average Sensitivity", xscale = log10)
plot_metric("LocalLipschitzEstimate", metric_name = "Local Lipschitz Estimate", xscale = log10)
plot_metric("GridPG", metric_name = "Grid Pointing Game", xscale = identity)
