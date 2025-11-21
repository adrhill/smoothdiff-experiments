using Pkg
Pkg.activate(@__DIR__)
using DrWatson
using DataFrames
using CairoMakie

include(joinpath(@__DIR__, "..", "theme.jl"))

const figdir = joinpath(@__DIR__, "figures")
!isdir(figdir) && mkdir(figdir)

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
    fig = Figure(; size = (LINE_WIDTH_NEURIPS, 5cm), figure_padding = 1mm)
    ax = Axis(fig[1, 1]; xlabel = "Number of samples", ylabel = "Walltime (s)")
    lines!(
        ax,
        df_sd[!, :n],
        df_sd[!, :time] / batchsize;
        label = "SmoothDiff",
        color = color_smoothdiff,
        linewidth = 2,
    )
    lines!(
        ax,
        df_sg[!, :n],
        df_sg[!, :time] / batchsize;
        label = "SmoothGrad",
        color = color_smoothgrad,
        linewidth = 2,

    )
    axislegend(ax; position = :rb, rowgap = -5, labelsize = 11)
    save(joinpath(figdir, "runtime.svg"), fig)
    display(fig)
end
