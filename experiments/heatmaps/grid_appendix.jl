using Pkg
Pkg.activate(@__DIR__)
using DrWatson

## Load dependencies
using CairoMakie
using Metalhead: VGG
using Flux: MaxPool, onecold
using VisionHeatmaps
using LinearAlgebra
using Statistics
using JLD2
using Colors
using StableRNGs
using Random

## Hotfix for Metalhead compatibility
import Flux: loadmodel!
loadmodel!(dst::MaxPool{N, M}, src::Tuple{}; kw...) where {N, M} = dst

include(joinpath(@__DIR__, "..", "theme.jl"))

figdir = joinpath(@__DIR__, "figures", "grid_appendix")
!isdir(figdir) && mkdir(figdir)

## Heatmapping
pipeline = NormReduction() |> PercentileClip() |> ExtremaColormap() |> FlipImage()
results_dir = joinpath(@__DIR__, "..", "cluster", "run_analyzers", "data")

const input_data = load(joinpath(@__DIR__, "..", "heatmaps", "data", "input_batch.jld2"))
const image_data = load(joinpath(@__DIR__, "..", "heatmaps", "data", "image_batch.jld2"))
const input = input_data["input"]
const targets = input_data["labels"]
const imgs = image_data["imgs"]

const expl_grad = load(joinpath(results_dir, "Gradient.jld2"))["val"]
const expl_sd10 = load(joinpath(results_dir, "SmoothDiff_n=10_std=0.5.jld2"))["val"]
const expl_sg10 = load(joinpath(results_dir, "SmoothGrad_n=10_std=0.5.jld2"))["val"]
const expl_conv = load(joinpath(results_dir, "SmoothGrad_n=100000_std=0.5.jld2"))["val"]
const expl_beta1 = load(joinpath(results_dir, "SoftPlusTrick_beta=1.0.jld2"))["val"]

const hs_grad = VisionHeatmaps.heatmap(expl_grad, pipeline)
const hs_sd10 = VisionHeatmaps.heatmap(expl_sd10, pipeline)
const hs_sg10 = VisionHeatmaps.heatmap(expl_sg10, pipeline)
const hs_conv = VisionHeatmaps.heatmap(expl_conv, pipeline)
const hs_beta1 = VisionHeatmaps.heatmap(expl_beta1, pipeline)

function add_img!(fig, img)
    ax = Axis(fig; aspect = DataAspect())
    image!(ax, rotr90(img); interpolate = false)
    hidedecorations!(ax)
    return ax
end

# Heuristic to add line breaks to overly long labels
function make_label(label)
    if length(label) > 15
        label = replace(label, " " => "\n")
    end
    return '"' * label * '"'
end

## Create grid
function plot_grid(i, idxs)
    fig = Figure(; size = (14cm / 2, 22.5cm))

    g = fig[1, 1] = GridLayout()

    function bg_box(loc, color)
        return Box(
            loc;
            color = alphacolor(color, 0.6),
            strokecolor = :white,
            cornerradius = 0,
            strokewidth = 0mm,
        )
    end
    bg_box(g[1, 1, Top()], Gray(0.7))
    bg_box(g[1, 2, Top()], color_gradient)
    bg_box(g[1, 3, Top()], color_smoothdiff)
    bg_box(g[1, 4, Top()], color_smoothgrad)
    bg_box(g[1, 5, Top()], color_smoothgrad)
    bg_box(g[1, 6, Top()], color_beta_smoothing)

    for (row, i) in enumerate(idxs)
        add_img!(g[row, 1], imgs[:, :, i])
        add_img!(g[row, 2], hs_grad[i])
        add_img!(g[row, 3], hs_sd10[i])
        add_img!(g[row, 4], hs_sg10[i])
        add_img!(g[row, 5], hs_conv[i])
        add_img!(g[row, 6], hs_beta1[i])
    end

    colgap!(g, 2)
    rowgap!(g, 2)

    ## Add heatmap labels
    imagenet_labels = readlines(
        download(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        ),
    )
    for (i, label) in enumerate(imagenet_labels[targets[idxs]])
        Label(
            g[i, 1, Left()],
            make_label(label);
            fontsize = 1.2mm,
            valign = :center,
            halign = :center,
            rotation = pi / 2,
            padding = (0, 0, 0, 0),
            tellheight = false,
        )
    end
    for (i, method) in enumerate(
            (
                "Input",
                "Gradient",
                "SmoothDiff\n(n=10)",
                "SmoothGrad\n(n=10)",
                "SmoothGrad\n(converged)",
                "β-Smoothing\n(β=1)",
            )
        )
        Label(
            g[1, i, Top()],
            method;
            fontsize = 1.5mm,
            valign = :center,
            halign = :center,
            padding = (0, 0, 1, 1),
        )
    end

    ## Save figure
    display(fig)
    save(joinpath(figdir, "grid_appendix_$i.png"), fig; px_per_unit = 600 / inch)
    return nothing
end

## Pick random images that are correctly classified
model = VGG(19; pretrain = true)
pred = onecold(model(input))
idxs = findall(pred .== targets)

# Remove samples from fig 1b for more diversity
idxs_fig1b = [4, 5, 7, 9, 13, 14, 18, 26]
idxs = filter!(i -> i ∉ idxs_fig1b, idxs)

# Randomly permute array
idxs = shuffle!(StableRNG(123), idxs)

# Make figures
nrows = 19

for (i, idxs) in enumerate(Iterators.partition(idxs, nrows))
    @info "Plotting grid" i idxs
    with_theme(theme_smoothdiff()) do
        plot_grid(i, idxs)
    end
end
