using Pkg
Pkg.activate(@__DIR__)
using DrWatson

## Load dependencies
using CairoMakie
using LaTeXStrings
using ImageFiltering
using LinearAlgebra
using Statistics
using JLD2
using Colors

include(joinpath(@__DIR__, "..", "theme.jl"))

figdir = joinpath(@__DIR__, "figures", "fig1a")
!isdir(figdir) && mkdir(figdir)

## Load results
res = load(datadir("1d_results.jld2"))
xs = res["inputs"]

## Compute convolution
kernel = Kernel.gaussian((res["std"] * length(xs) / (maximum(xs) - minimum(xs)),))
convolution = imfilter(res["gradient"], kernel)

## Draw figure
with_theme(theme_smoothdiff()) do
    fig_1d = Figure(; size = (LINE_WIDTH_NEURIPS, 4.75cm), figure_padding = (0mm, 0mm, 2mm, 2mm))

    ## 1D Example
    ax_1d = Axis(fig_1d[1, 1]; xlabel = "Input")

    # Bands
    myalpha = 0.4
    band!(
        ax_1d,
        xs,
        res["smoothgrad_q5"],
        res["smoothgrad_q95"];
        color = color_smoothgrad,
        alpha = myalpha,
    )
    band!(
        ax_1d,
        xs,
        res["smoothdiff_q5"],
        res["smoothdiff_q95"];
        color = color_smoothdiff,
        alpha = myalpha,
    )

    # Lines
    lines!(ax_1d, xs, res["gradient"]; label = L"Gradient", color = color_gradient)
    lines!(ax_1d, xs, res["softplus"]; label = L"$β$-Smoothing", color = color_beta_smoothing)
    lines!(ax_1d, xs, convolution; linewidth = 2, label = "Convolution", color = color_convolution)
    lines!(ax_1d, xs, res["smoothgrad_mean"]; label = "SmoothGrad", color = color_smoothgrad)
    lines!(ax_1d, xs, res["smoothdiff_mean"]; label = "SmoothDiff", color = color_smoothdiff)

    # Add legend
    elems = [
        LineElement(; color = color_gradient),
        [
            PolyElement(; color = color_smoothdiff, alpha = myalpha),
            LineElement(; color = color_smoothdiff),
        ],
        [
            PolyElement(; color = color_smoothgrad, alpha = myalpha),
            LineElement(; color = color_smoothgrad),
        ],
        LineElement(; color = color_beta_smoothing),
        LineElement(; color = color_convolution),
    ]
    Legend(
        fig_1d[1, 0],
        elems,
        ["Gradient", "SmoothDiff", "SmoothGrad", "β-Smoothing", "Convolution"];
        orientation = :vertical,
        framevisible = false,
    )
    Label(
        fig_1d[1, 0, Top()],
        "(a)";
        color = :black,
        fontsize = 4mm,
        font = :bold,
        valign = :center,
        halign = :left,
        padding = (0, 0, 0, 0),
        tellheight = false,
        tellwidth = false,
    )
    save(joinpath(figdir, "fig1a.svg"), fig_1d)
    display(fig_1d)
end
