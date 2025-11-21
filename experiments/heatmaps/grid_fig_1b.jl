using Pkg
Pkg.activate(@__DIR__)
using DrWatson

## Load dependencies
using CairoMakie
using VisionHeatmaps
using LinearAlgebra
using Statistics
using JLD2
using Colors

include(joinpath(@__DIR__, "..", "theme.jl"))

figdir = joinpath(@__DIR__, "figures", "fig1b")
!isdir(figdir) && mkdir(figdir)

## Heatmapping
pipeline = NormReduction() |> PercentileClip() |> ExtremaColormap() |> FlipImage()
results_dir = joinpath(@__DIR__, "..", "cluster", "run_analyzers", "data")

const input_data = load(joinpath(@__DIR__, "..", "heatmaps", "data", "input_batch.jld2"))
const image_data = load(joinpath(@__DIR__, "..", "heatmaps", "data", "image_batch.jld2"))
const input = input_data["input"]
const labels = input_data["labels"]
const imgs = image_data["imgs"]

const expl_grad = load(joinpath(results_dir, "Gradient.jld2"))["val"]
const expl_sd10 = load(joinpath(results_dir, "SmoothDiff_n=10_std=0.5.jld2"))["val"]
const expl_sg10 = load(joinpath(results_dir, "SmoothGrad_n=10_std=0.5.jld2"))["val"]
const expl_conv = load(joinpath(results_dir, "SmoothGrad_n=100000_std=0.5.jld2"))["val"]

const hs_grad = VisionHeatmaps.heatmap(expl_grad, pipeline)
const hs_sd10 = VisionHeatmaps.heatmap(expl_sd10, pipeline)
const hs_sg10 = VisionHeatmaps.heatmap(expl_sg10, pipeline)
const hs_conv = VisionHeatmaps.heatmap(expl_conv, pipeline)

function add_img!(f, img)
    ax = Axis(f; aspect = DataAspect())
    image!(ax, rotr90(img); interpolate = false)
    hidedecorations!(ax)
    return ax
end

## Add heatmaps #
idxs = [4, 5, 7, 9, 13, 14, 18, 26]
ncols = length(idxs)

with_theme(theme_smoothdiff()) do
    fig = Figure(; size = (14cm, 8.5cm))
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
    bg_box(g[2, 1, Left()], color_gradient)
    bg_box(g[3, 1, Left()], color_smoothdiff)
    bg_box(g[4, 1, Left()], color_smoothgrad)
    bg_box(g[5, 1, Left()], color_smoothgrad)

    for (col, i) in enumerate(idxs)
        add_img!(g[1, col], imgs[:, :, i])
        add_img!(g[2, col], hs_grad[i])
        add_img!(g[3, col], hs_sd10[i])
        add_img!(g[4, col], hs_sg10[i])
        add_img!(g[5, col], hs_conv[i])
    end

    colgap!(g, 2)
    rowgap!(g, 2)

    ## Add heatmap labels
    # Target labels:  278  844  425  752  967  418  753  306
    # Predictions:    278  844  425  752  967  418  753  306
    for (i, label) in enumerate(
            (
                "red fox",
                "swing",
                "barbershop",
                "race car",
                "red wine",
                "balloon",
                "racket",
                "dung beetle",
            )
        )
        Label(
            g[1, i, Top()],
            """\"$label\"""";
            fontsize = 2.5mm,
            valign = :center,
            halign = :center,
        )
    end
    for (i, method) in enumerate(
            (
                "Input",
                "Gradient",
                "SmoothDiff\n(n=10)",
                "SmoothGrad\n(n=10)",
                "SmoothGrad\n(converged)",
            )
        )
        Label(
            g[i, 1, Left()],
            method;
            fontsize = 2.5mm,
            valign = :center,
            halign = :center,
            rotation = pi / 2,
            padding = (1.5, 1.5, 0, 0),
            tellheight = false,
        )
    end

    Label(
        g[1, 1, TopLeft()],
        "(b)";
        fontsize = 4mm,
        font = :bold,
        valign = :center,
        halign = :center,
        padding = (0, 2, 0, 0),
        tellheight = false,
        tellwidth = false,
    )

    ## Save figure
    save(joinpath(figdir, "fig1b.png"), fig; px_per_unit = 600 / inch)
    display(fig)
end
