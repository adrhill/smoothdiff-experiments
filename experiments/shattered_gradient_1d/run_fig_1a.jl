using Pkg
Pkg.activate(@__DIR__)
using DrWatson

## Import dependencies
using Distributions
using ExplainableAI
using Flux
using SmoothedDifferentiation
using LinearAlgebra
using Statistics: mean
using Random
using StableRNGs
using JLD2

## Compute 1D example
process_input(x) = [x;;]
process_output(expl) = only(expl.val)

compute_1d(analyzer, xs::AbstractVector) = [compute_1d(analyzer, x) for x in xs]
function compute_1d(analyzer, x::Real)
    expl = analyze([x;;], analyzer)
    return only(expl.val)
end

function run_all()
    # Initialize model weights randomly like in the shattered gradient paper
    n_neurons = 64
    n_layers = 16
    model = Chain(
        Dense(1, n_neurons, relu),
        repeat([Dense(n_neurons, n_neurons, relu)], n_layers - 2)...,
        Dense(n_neurons, 1),
    )

    rng = StableRNG(12345)
    for l in model.layers
        randn!(rng, l.weight)
        randn!(rng, l.bias)
        n_bias = sqrt(length(l.bias))
        l.weight ./= n_bias
        l.bias ./= n_bias
    end

    # Define analyzers
    n_samples = 10
    std = 0.05f0
    distr = Normal(0.0f0, std)
    beta = Float32(2 * log(2) * sqrt(2 * pi / std^2) * n_layers)

    analyzer_gradient = Gradient(model)
    analyzer_sg = SmoothGrad(model, n_samples, distr)
    analyzer_sd = SmoothDiff(model, [1.0f0;;], n_samples, distr)
    analyzer_softplus = SoftPlusTrick(model, beta)

    # Run analyzers
    input_range = range(-1.0f0, 1.0f0; length = 1001)
    n_runs = 2^10 # used for stochastic analyzers

    val_gradient = compute_1d(analyzer_gradient, input_range)
    val_softplus = compute_1d(analyzer_softplus, input_range)

    @info "Running SmoothDiff..."
    vals_sd = hcat([compute_1d(analyzer_sd, input_range) for _ in 1:n_runs]...)
    val_sd_mean = vec(mean(vals_sd; dims = 2))
    val_sd_max = vec(maximum(vals_sd; dims = 2))
    val_sd_min = vec(minimum(vals_sd; dims = 2))
    val_sd_q5 = quantile.(eachrow(vals_sd), 0.05)
    val_sd_q95 = quantile.(eachrow(vals_sd), 0.95)

    @info "Running SmoothGrad..."
    vals_sg = hcat([compute_1d(analyzer_sg, input_range) for _ in 1:n_runs]...)
    val_sg_mean = vec(mean(vals_sg; dims = 2))
    val_sg_max = vec(maximum(vals_sg; dims = 2))
    val_sg_min = vec(minimum(vals_sg; dims = 2))
    val_sg_q5 = quantile.(eachrow(vals_sg), 0.05)
    val_sg_q95 = quantile.(eachrow(vals_sg), 0.95)

    # Save results
    jldsave(
        datadir("1d_results.jld2");
        inputs = collect(input_range),
        gradient = val_gradient,
        softplus = val_softplus,
        smoothgrad_mean = val_sg_mean,
        smoothgrad_max = val_sg_max,
        smoothgrad_min = val_sg_min,
        smoothgrad_q5 = val_sg_q5,
        smoothgrad_q95 = val_sg_q95,
        smoothdiff_mean = val_sd_mean,
        smoothdiff_max = val_sd_max,
        smoothdiff_min = val_sd_min,
        smoothdiff_q5 = val_sd_q5,
        smoothdiff_q95 = val_sd_q95,
        std = std,
        n_samples = n_samples,
        beta = beta,
    )
    return nothing
end

run_all()
