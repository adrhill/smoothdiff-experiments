using Pkg
Pkg.activate(@__DIR__)
using DrWatson
using DataFrames, JLD2, PrettyTables
using CairoMakie

include(joinpath(@__DIR__, "..", "theme.jl"))

const figdir = joinpath(@__DIR__, "figures")
!isdir(figdir) && mkdir(figdir)

## Load computed results
const results_dir = joinpath(
    @__DIR__, "..", "cluster", "pixelflipping", "data", "run_analyzers"
)

function method_name(method, n, std)
    if method == "SmoothDiff" || method == "InfiniGrad"
        return "SmoothDiff (n=$n, σ=$std)"
    elseif method == "SmoothGrad"
        return "SmoothGrad (n=$n, σ=$std)"
    else
        return method
    end
end

df_raw = collect_results(results_dir)
df_full = transform(
    df_raw,
    :LIF => ByRow(x -> (sum(x) / (length(x) - 1))) => :LIF_area,
    :MIF => ByRow(x -> (sum(x) / (length(x) - 1))) => :MIF_area,
    [:method, :n, :std] => ByRow(method_name) => :name,
    :path => ByRow(x -> last(splitpath(x))[4:(end - 4)]) => :pathname,
)
sort!(df_full, [:method, order(:n; rev = false)])

df_nmax = filter(r -> r.n <= 2^6, df_full)
df_sd = filter(row -> row.method == "SmoothDiff" || row.method == "InfiniGrad" && row.std == 0.5, df_nmax)
df_sg = filter(row -> row.method == "SmoothGrad" && row.std == 0.5, df_nmax)
sort!(df_sd, :n; rev = false)
sort!(df_sg, :n; rev = false)

with_theme(theme_smoothdiff()) do
    fig = Figure(; size = (LINE_WIDTH_NEURIPS, 4cm), figure_padding = 1mm)
    ax = Axis(fig[1, 1]; xlabel = "Number of samples", ylabel = "SRG")
    lines!(ax, df_sd[:, :n], df_sd[:, :SRG]; label = "SmoothDiff")
    lines!(ax, df_sg[:, :n], df_sg[:, :SRG]; label = "SmoothGrad")
    axislegend(ax; position = :rb, rowgap = -5, labelsize = 11)

    save(joinpath(figdir, "srg_over_n.svg"), fig)
    display(fig)
end

## Print table
df_table = select(df_full, :pathname, :LIF_area, :MIF_area, :SRG)
sort!(df_table, :SRG; rev = true)
pretty_table(df_table)
