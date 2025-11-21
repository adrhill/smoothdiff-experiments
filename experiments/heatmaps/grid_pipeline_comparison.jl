using Pkg
Pkg.activate(@__DIR__)
using DrWatson

## Load dependencies
using CairoMakie
using VisionHeatmaps
using ImageFiltering
using ColorSchemes

using LinearAlgebra
using Statistics
using JLD2
using Colors

include(joinpath(@__DIR__, "..", "theme.jl"))

figdir = joinpath(@__DIR__, "figures", "grid_pipelines")
!isdir(figdir) && mkdir(figdir)

## Define smoothing transform
import VisionHeatmaps: AbstractTransform, apply

struct Smoothing{T} <: AbstractTransform
    kernel::T
end

function apply(t::Smoothing, x, _img)
    return imfilter(x, t.kernel)
end

## Heatmapping
pipeline1 = NormReduction() |> PercentileClip() |> ExtremaColormap() |> FlipImage()
pipeline2 =
    NormReduction() |>
    Smoothing(Kernel.gaussian(10)) |>
    PercentileClip() |>
    ExtremaColormap(:jet) |>
    FlipImage() |>
    AlphaOverlay(0.5)

results_dir = joinpath(@__DIR__, "..", "cluster", "run_analyzers", "data")

const input_data = load(joinpath(@__DIR__, "..", "heatmaps", "data", "input_batch.jld2"))
const image_data = load(joinpath(@__DIR__, "..", "heatmaps", "data", "image_batch.jld2"))
const input = input_data["input"]
const labels = input_data["labels"]
const imgs = image_data["imgs"]

const expl = load(joinpath(results_dir, "SmoothDiff_n=10_std=0.5.jld2"))["val"]

const hs1 = VisionHeatmaps.heatmap(expl, pipeline1)
const hs2 = VisionHeatmaps.heatmap(expl, pipeline2)

function add_img!(fig, img)
    ax = Axis(fig; aspect = DataAspect())
    image!(ax, rotr90(img); interpolate = false)
    hidedecorations!(ax)
    return ax
end

## Add heatmaps #
idxs = [4, 5, 7, 9, 13, 14, 18, 26]
ncols = length(idxs)
nrows = 5

function bg_box(loc, color)
    return Box(
        loc;
        color = alphacolor(color, 0.6),
        strokecolor = :white,
        cornerradius = 0,
        strokewidth = 0mm,
    )
end

with_theme(theme_smoothdiff()) do
    fig = Figure(; size = (14cm, 5.25cm))
    g = fig[1, 1] = GridLayout()

    bg_box(g[1, 1, Left()], RGB(1, 1, 1))
    bg_box(g[2, 1, Left()], color_smoothdiff)
    bg_box(g[3, 1, Left()], color_smoothdiff)

    for (col, i) in enumerate(idxs)
        img = imgs[:, :, i]
        img_gray = convert.(Gray, img)
        h1 = only(VisionHeatmaps.heatmap(expl[:, :, :, i:i], img_gray, pipeline1))
        h2 = only(VisionHeatmaps.heatmap(expl[:, :, :, i:i], img_gray, pipeline2))

        add_img!(g[1, col], img)
        add_img!(g[2, col], h1)
        add_img!(g[3, col], h2)
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
    for (i, method) in enumerate(("Input", "Batlow", "Jet\n(overlay)"))
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

    ## Save figure
    save(joinpath(figdir, "grid_pipelines.png"), fig; px_per_unit = 600 / inch)
    display(fig)
end
