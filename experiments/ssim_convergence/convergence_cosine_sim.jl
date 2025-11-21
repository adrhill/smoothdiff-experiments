using Pkg
Pkg.activate(@__DIR__)
using DrWatson
using DataFrames

using LinearAlgebra
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

## Use cosine similarity as a metric for convergence
function cosine_similarity(A::AbstractArray, B::AbstractArray)
    A, B = vec(A), vec(B)
    return dot(A, B) / (norm(A) * norm(B))
end

function cosine_similarity_batch(b1, b2)
    return [
        cosine_similarity(s1, s2) for
            (s1, s2) in Iterators.zip(eachslice(b1; dims = 4), eachslice(b2; dims = 4))
    ]
end
cosine_similarity_sg(n) = cosine_similarity_batch(expl_sg(n), expl_ref)
cosine_similarity_sd(n) = cosine_similarity_batch(expl_sd(n), expl_ref)

## Relative magnitude
function relative_magnitude(A::AbstractArray, B::AbstractArray)
    mA = norm(vec(A))
    mB = norm(vec(B))
    return mA / mB
end

function relative_magnitude_batch(b1, b2)
    return [
        relative_magnitude(s1, s2) for
            (s1, s2) in Iterators.zip(eachslice(b1; dims = 4), eachslice(b2; dims = 4))
    ]
end
relative_magnitude_sg(n) = relative_magnitude_batch(expl_sg(n), expl_ref)
relative_magnitude_sd(n) = relative_magnitude_batch(expl_sd(n), expl_ref)

ns_log = 2 .^ (1:11)
ns_lin = 5:5:50
ns = sort!(union(ns_log..., ns_lin...))
trunc_ns = filter(<=(100), ns)
n_trunc_ns = length(trunc_ns)

# allocate outputs
batchsize = size(expl_ref, 4) # WHCN format
cs_sd = Matrix{Float32}(undef, length(ns), batchsize)
cs_sg = Matrix{Float32}(undef, length(ns), batchsize)
mag_sd = Matrix{Float32}(undef, length(ns), batchsize)
mag_sg = Matrix{Float32}(undef, length(ns), batchsize)

# compute similarities
for (i, n) in enumerate(ns)
    cs_sd[i, :] .= cosine_similarity_sd(n)
    cs_sg[i, :] .= cosine_similarity_sg(n)
    mag_sd[i, :] .= relative_magnitude_sd(n)
    mag_sg[i, :] .= relative_magnitude_sg(n)
end

## Compute means
means_cs_sd = vec(mean(cs_sd; dims = 2))
means_cs_sg = vec(mean(cs_sg; dims = 2))
means_mag_sd = vec(mean(mag_sd; dims = 2))
means_mag_sg = vec(mean(mag_sg; dims = 2))

with_theme(theme_smoothdiff()) do
    fig = Figure(; size = (LINE_WIDTH_NEURIPS, 7cm), figure_padding = 1mm)
    ax_cos = Axis(
        fig[1, 1];
        ylabel = "Cosine Similarity",
        yticks = 0.4:0.2:1.0,
    )
    ax_mag = Axis(
        fig[2, 1];
        xlabel = "Number of samples",
        ylabel = "Relative Magnitude",
    )

    # Plot cosine similarities
    l_sd = lines!(
        ax_cos,
        trunc_ns,
        means_cs_sd[1:n_trunc_ns];
        label = "SmoothDiff",
        color = color_smoothdiff,
        linewidth = 2,
    )
    l_sg = lines!(
        ax_cos,
        trunc_ns,
        means_cs_sg[1:n_trunc_ns];
        label = "SmoothGrad",
        color = color_smoothgrad,
        linewidth = 2,
    )

    # Plot magnitudes
    lines!(
        ax_mag,
        trunc_ns,
        means_mag_sd[1:n_trunc_ns];
        label = "SmoothDiff",
        color = color_smoothdiff,
        linewidth = 2,
    )
    lines!(
        ax_mag,
        trunc_ns,
        means_mag_sg[1:n_trunc_ns];
        label = "SmoothGrad",
        color = color_smoothgrad,
        linewidth = 2,
    )

    ## Align axes and make plot pretty
    linkxaxes!(ax_cos, ax_mag)
    ylims!(ax_cos; high = 1.0)
    ylims!(ax_mag; low = 0.0)
    yspace = maximum(tight_yticklabel_spacing!, [ax_cos, ax_mag])
    ax_cos.yticklabelspace = yspace
    ax_mag.yticklabelspace = yspace

    ## Add legend and save
    Legend(fig[1, :, Top()], [l_sd, l_sg], ["SmoothDiff", "SmoothGrad"], orientation = :horizontal, labelsize = 11)

    save(joinpath(figdir, "convergence_cosine_sim.svg"), fig)
    display(fig)
end
