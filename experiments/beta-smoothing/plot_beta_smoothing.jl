using Pkg
Pkg.activate(@__DIR__)
using DrWatson

using CairoMakie
using NNlib

using SpecialFunctions
using DifferentiationInterface
import ForwardDiff
backend = AutoForwardDiff()

include(joinpath(@__DIR__, "..", "theme.jl"))

const figdir = joinpath(@__DIR__, "figures")
!isdir(figdir) && mkdir(figdir)

xs = range(-4, 4, length = 501)

σ_default = 1.0
g(x, σ = σ_default) = 1 / sqrt(2 * pi) / σ * exp(-1 / 2 * (x / σ)^2) # Gaussian kernel
smoothheavy(x, σ = σ_default) = 1 / 2 * (1 + erf(x / sqrt(2) / σ))
smoothrelu(x, σ = σ_default) = x * smoothheavy(x, σ) + σ^2 * g(x, σ)

# Compute derivatives with AD
prep_relu = prepare_derivative(relu, backend, 1.0)
der_relu = [derivative(relu, prep_relu, backend, x) for x in xs]

prep_softplus = prepare_derivative(softplus, backend, 1.0)
der_softplus = [derivative(softplus, prep_softplus, backend, x) for x in xs]

prep_smoothrelu = prepare_derivative(smoothrelu, backend, 1.0)
der_smoothrelu = [derivative(smoothrelu, prep_smoothrelu, backend, x) for x in xs]

## Draw figure
with_theme(theme_smoothdiff()) do
    fig = Figure(; size = (LINE_WIDTH_NEURIPS, 4cm), figure_padding = 1mm)

    ax1 = Axis(fig[2, 1], yticks = 0:2:4, xlabel = "x", ylabel = "f(x)")
    l13 = lines!(ax1, xs, softplus(xs), color = color_beta_smoothing)
    l12 = lines!(ax1, xs, smoothrelu.(xs), color = color_smoothdiff)
    l11 = lines!(ax1, xs, relu(xs), color = color_gradient)

    ax2 = Axis(fig[2, 2]; xlabel = "x", ylabel = "f'(x)")
    l23 = lines!(ax2, xs, der_softplus, color = color_beta_smoothing)
    l22 = lines!(ax2, xs, smoothheavy.(xs), color = color_smoothdiff)
    l21 = lines!(ax2, xs, der_relu, color = color_gradient)

    Legend(fig[1, :], [l11, l12, l13], ["ReLU", "Smoothed ReLU (σ=1)", "Softplus (β=1)"], orientation = :horizontal, labelsize = 11, position = :rb)
    rowgap!(fig.layout, 1, 0.1cm)

    save(joinpath(figdir, "beta_smoothing.svg"), fig)
    fig
end
