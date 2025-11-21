using Pkg
Pkg.activate(@__DIR__)
using DrWatson
using DataFrames

using ImageQualityIndexes
using VisionHeatmaps

using Statistics
using JLD2
using CairoMakie

include(joinpath(@__DIR__, "..", "theme.jl"))

const figdir = joinpath(@__DIR__, "figures")
!isdir(figdir) && mkdir(figdir)

const results_dir = joinpath(@__DIR__, "..", "cluster", "run_analyzers", "data")
const benchmark_dir = joinpath(
    @__DIR__, "..", "cluster", "run_analyzers", "data", "benchmarks"
)
const pipeline = NormReduction() |> PercentileClip() |> ExtremaColormap() |> FlipImage()
const expl_ref = load(joinpath(results_dir, "SmoothGrad_n=100000_std=0.5.jld2"))["val"]

const heatmaps_ref = VisionHeatmaps.heatmap(expl_ref, pipeline)

## Load explanation
const std = 0.5
expl_sg(n) = load(joinpath(results_dir, "SmoothGrad_n=$(n)_std=$(std).jld2"))["val"]
expl_sd(n) = load(joinpath(results_dir, "SmoothDiff_n=$(n)_std=$(std).jld2"))["val"]
heatmaps_sg(n) = VisionHeatmaps.heatmap(expl_sg(n), pipeline)
heatmaps_sd(n) = VisionHeatmaps.heatmap(expl_sd(n), pipeline)

## Use Structural Similarity Index (SSIM)  as a metric for convergence
function image_similarity_batch(hs1, hs2)
    return [assess_ssim(h1, h2) for (h1, h2) in Iterators.zip(hs1, hs2)]
end
image_similarity_sg(n) = image_similarity_batch(heatmaps_sg(n), heatmaps_ref)
image_similarity_sd(n) = image_similarity_batch(heatmaps_sd(n), heatmaps_ref)

ns_log = 2 .^ (1:11)
ns_lin = 5:5:50
ns = sort!(union(ns_log..., ns_lin...))
trunc_ns = filter(<=(100), ns)
n_trunc_ns = length(trunc_ns)

# allocate outputs
batchsize = size(expl_ref, 4) # WHCN format
cs_sd = Matrix{Float32}(undef, length(ns), batchsize)
cs_sg = Matrix{Float32}(undef, length(ns), batchsize)
cs_sd_sd = Matrix{Float32}(undef, length(ns), batchsize)

# compute similarities
for (i, n) in enumerate(ns)
    cs_sd[i, :] .= image_similarity_sd(n)
    cs_sg[i, :] .= image_similarity_sg(n)
    cs_sd_sd[i, :] .= image_similarity_batch(heatmaps_sd(n), heatmaps_sd(2^11))
end

## Compute means
means_sd = vec(mean(cs_sd; dims = 2))
means_sg = vec(mean(cs_sg; dims = 2))


## Load benchmarks
benchmark_dir = joinpath(@__DIR__, "..", "cluster", "run_analyzers", "data", "benchmarks")
df_full = collect_results(benchmark_dir)

n_max = 64 # max amount of samples
batchsize = 128

df_sd = filter(
    row -> row.method == "SmoothDiff" || row.method == "InfiniGrad" && row.std == 0.5
        && row.n <= n_max, df_full
)
df_sg = filter(row -> row.method == "SmoothGrad" && row.std == 0.5 && row.n <= n_max, df_full)
sort!(df_sd, :n, rev = false)
sort!(df_sg, :n, rev = false)

n_first = 21 # only plot up to n_first

with_theme(theme_smoothdiff()) do
    ## Plot SSIM convergence
    fig = Figure(; size = (LINE_WIDTH_NEURIPS, 4cm), figure_padding = 1mm)
    ax_ssim = Axis(
        fig[2, 1];
        xlabel = "Number of samples",
        ylabel = "Mean SSIM",
    )
    lines!(
        ax_ssim,
        trunc_ns,
        means_sd[1:n_trunc_ns];
        label = "SmoothDiff",
        color = color_smoothdiff,
        linewidth = 2,
    )
    lines!(
        ax_ssim,
        trunc_ns,
        means_sg[1:n_trunc_ns];
        label = "SmoothGrad",
        color = color_smoothgrad,
        linewidth = 2,
    )

    ## Draw arrow with text
    n_sg = 50
    ssim = means_sg[findfirst(==(n_sg), ns)]
    n_sd = 10
    arrows2d!(
        ax_ssim,
        [n_sg],
        [ssim],
        [n_sd - n_sg],
        [0];
        color = _wong_colors[3],
        shaftwidth = 2.5,
        tipwidth = 10,
    )
    text!(
        ax_ssim,
        n_sd + (n_sg - n_sd) / 2,
        ssim + 0.05;
        align = (:center, :bottom),
        text = "×0.2",
        color = _wong_colors[3],
        fontsize = 13,
        font = :bold,
    )

    ## Add vertical line
    vlines!(ax_ssim, 50; color = colorant"#777777", label = "n=50")
    ylims!(ax_ssim, 0.4, 1.0)

    ## Plot runtime benchmarks
    ax_runtime = Axis(fig[2, 2]; xlabel = "Number of samples", ylabel = "Walltime (s)")
    lines!(
        ax_runtime,
        df_sd[!, :n],
        df_sd[!, :time] / batchsize;
        label = "SmoothDiff",
        color = color_smoothdiff,
        linewidth = 2,
    )
    lines!(
        ax_runtime,
        df_sg[!, :n],
        df_sg[!, :time] / batchsize;
        label = "SmoothGrad",
        color = color_smoothgrad,
        linewidth = 2,
    )
    colsize!(fig.layout, 2, 4cm)

    ## Add legend
    Legend(
        fig[1, :],
        ax_runtime,
        orientation = :horizontal,
        patchsize = (30, 10),
        labelsize = 11,
    )
    rowsize!(fig.layout, 1, 0.1cm)
    rowgap!(fig.layout, 1, 0.3cm)

    save(joinpath(figdir, "performance.svg"), fig)
    save(joinpath(figdir, "performance.png"), fig, px_per_unit = 300 / inch)
    display(fig)
end
