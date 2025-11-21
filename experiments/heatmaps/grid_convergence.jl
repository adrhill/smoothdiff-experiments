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

const figdir = joinpath(@__DIR__, "figures", "grid_convergence")
!isdir(figdir) && mkdir(figdir)

const results_dir = joinpath(@__DIR__, "..", "cluster", "run_analyzers", "data")
const pipeline = NormReduction() |> PercentileClip() |> ExtremaColormap() |> FlipImage()

const input_data = load(datadir("input_batch.jld2")) # from save_input.jl
const image_data = load(datadir("image_batch.jld2")) # from save_input.jl
const input = input_data["input"]
const labels = input_data["labels"]
const imgs = image_data["imgs"]

## Load computed results
std = 0.5
function expl_sg(n)
    return load(joinpath(results_dir, "SmoothGrad_n=$(n)_std=$(std).jld2"))["val"]
end
function expl_sd(n)
    return load(joinpath(results_dir, "SmoothDiff_n=$(n)_std=$(std).jld2"))["val"]
end

expl_grad = load(joinpath(results_dir, "Gradient.jld2"))["val"]
h_grad = VisionHeatmaps.heatmap(expl_grad, pipeline)

df_full = collect_results(results_dir)
df = filter(:std => s -> s == std, df_full)
ns = 2 .^ (2:7)

nrows = 2
ncols = length(ns) + 1

## Compute heatmap of input i
function heatmap_experiment(row::DataFrameRow, i)
    return VisionHeatmaps.heatmap(row.val, pipeline)[i]
end

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
    fig = Figure(; size = (14cm, 4cm))

    # Draw heatmap grid
    g = fig[1, 1] = GridLayout()
    for (j, n) in enumerate(ns)
        h_sg = VisionHeatmaps.heatmap(expl_sg(n), pipeline)
        h_sd = VisionHeatmaps.heatmap(expl_sd(n), pipeline)

        add_img!(g[1, j], h_sd[idx])
        add_img!(g[2, j], h_sg[idx])
    end
    add_img!(g[1, ncols], imgs[:, :, idx])
    add_img!(g[2, ncols], h_grad[idx])

    function bg_box(loc, color)
        return Box(
            loc;
            color = alphacolor(color, 0.6),
            strokecolor = :white,
            cornerradius = 0,
            strokewidth = 0mm,
        )
    end
    bg_box(g[1, 1, Left()], color_smoothdiff)
    bg_box(g[2, 1, Left()], color_smoothgrad)

    bg_box(g[1, ncols, Left()], Gray(0.7))
    bg_box(g[2, ncols, Left()], color_gradient)

    colgap!(g, 2)
    rowgap!(g, 2)
    colgap!(g, ncols - 1, 7)

    for (i, label) in enumerate(("SmoothDiff", "SmoothGrad"))
        Label(
            g[i, 1, Left()],
            label;
            fontsize = 2.5mm,
            valign = :center,
            halign = :center,
            rotation = pi / 2,
            padding = (1.5, 1.5, 0, 0),
            tellheight = false,
        )
    end
    for (i, label) in enumerate(("Input", "Gradient"))
        Label(
            g[i, ncols, Left()],
            label;
            fontsize = 2.5mm,
            valign = :center,
            halign = :center,
            rotation = pi / 2,
            padding = (1.5, 1.5, 0, 0),
            tellheight = false,
        )
    end

    for (i, n) in enumerate(ns)
        Label(
            g[1, i, Top()],
            "n=$(n)";
            fontsize = 2.5mm,
            valign = :bottom,
            halign = :center,
        )
    end
    Label(
        g[1, ncols, Top()],
        """'$label'""";
        fontsize = 2.5mm,
        valign = :bottom,
        halign = :center,
    )
    save(joinpath(figdir, "grid_convergence_$(label).png"), fig; px_per_unit = 600 / inch)
    display(fig)
end
