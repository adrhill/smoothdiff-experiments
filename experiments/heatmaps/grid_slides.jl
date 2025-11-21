using Pkg
Pkg.activate(@__DIR__)
using DrWatson

using DataFrames, JLD2
using VisionHeatmaps
using ImageInTerminal
using FileIO, ImageIO
using CairoMakie
using Colors

include(joinpath(@__DIR__, "..", "theme.jl"))
set_theme!(theme_smoothdiff())

const figdir = joinpath(@__DIR__, "figures", "grid_slides")
!isdir(figdir) && mkdir(figdir)

const results_dir = joinpath(@__DIR__, "..", "cluster", "run_analyzers", "data")
const pipeline = NormReduction() |> PercentileClip() |> ExtremaColormap() |> FlipImage()

const input_data = load(datadir("input_batch.jld2")) # from save_input.jl
const image_data = load(datadir("image_batch.jld2")) # from save_input.jl
const input = input_data["input"]
const labels = input_data["labels"]
const imgs = image_data["imgs"]

## Load computed results
expl_grad = load(joinpath(results_dir, "Gradient.jld2"))["val"]
expl_sd = load(joinpath(results_dir, "SmoothDiff_n=20_std=0.5.jld2"))["val"]

h_grad = VisionHeatmaps.heatmap(expl_grad, pipeline)
h_sd = VisionHeatmaps.heatmap(expl_sd, pipeline)

## Add heatmap image to Makie plot
function add_img!(fig, img)
    ax = Axis(fig; aspect = DataAspect(), alignmode = Outside())
    image!(ax, rotr90(img); interpolate = false)
    hidedecorations!(ax)
    return ax
end

samples = Dict(
    4 => "red fox",
    5 => "swing",
    7 => "barbershop",
    9 => "race car",
    13 => "red wine",
    14 => "balloon",
    18 => "racket",
    26 => "dung beetle",
)
for (idx, label) in samples
    ## Draw grid figure
    fig = Figure(; size = (8.2cm, 2cm))

    # Draw heatmap grid
    g = fig[1, 1] = GridLayout()
    add_img!(g[1, 1], imgs[:, :, idx])
    add_img!(g[1, 2], h_grad[idx])
    add_img!(g[1, 3], h_sd[idx])

    function bg_box(loc, color)
        return Box(
            loc;
            color = alphacolor(color, 0.6),
            strokecolor = :white,
            cornerradius = 0,
            strokewidth = 0mm,
        )
    end

    bg_box(g[1, 1, Left()], Gray(0.7))
    bg_box(g[1, 2, Left()], color_gradient)
    bg_box(g[1, 3, Left()], color_smoothdiff)

    colgap!(g, 20)

    for (i, label) in enumerate(("Input", "Gradient", "SmoothDiff"))
        Label(
            g[1, i, Left()],
            label;
            fontsize = 2.5mm,
            valign = :center,
            halign = :center,
            rotation = pi / 2,
            padding = (1.5, 1.5, 0, 0),
            tellheight = false,
        )
    end

    save(joinpath(figdir, "grid_slides_$(label).png"), fig; px_per_unit = 600 / inch)
    display(fig)
end
