module SmoothedDifferentiation

using Reexport
@reexport using XAIBase
import XAIBase: call_analyzer

using Base.Iterators
using Distributions: Distribution, Sampleable, Normal
using Random: AbstractRNG, GLOBAL_RNG, rand!
using ProgressMeter: Progress, next!

using NNlib: relu, ∇maxpool, maxpool, upsample_nearest, σ, softplus
using Zygote: pullback
import ChainRulesCore: rrule, NoTangent

using Flux: Flux

include("prepare_model.jl")
include("vejp/relu.jl")
include("vejp/maxpool.jl")
include("softplus_trick.jl")

export SmoothDiff, SoftPlusTrick

## Helpers for Flux & Lux compatibility
mytestmode!(model) = model
mytestmode!(model::Flux.Chain) = Flux.testmode!(model)


samplingmode!(model, mode::Bool) = foreach(x -> samplingmode!(x, mode), Flux.trainable(model))

## Interface
const DEFAULT_SAMPLES = 50
const DEFAULT_DISTR = Normal(0.0f0, 1.0f0)

struct SmoothDiff{M, D <: Sampleable, R <: AbstractRNG} <: AbstractXAIMethod
    model::M
    n::Int
    distribution::D
    rng::R
    show_progress::Bool

    function SmoothDiff(
            model,
            input,
            n::Int = DEFAULT_SAMPLES,
            distribution::D = DEFAULT_DISTR,
            rng::R = GLOBAL_RNG,
            show_progress = true,
        ) where {D <: Sampleable, R <: AbstractRNG}
        prepared_model = prepare(model, input)
        mytestmode!(prepared_model)
        return new{typeof(prepared_model), D, R}(
            prepared_model, n, distribution, rng, show_progress
        )
    end
end

# Default to sampling from Normal distribution
function SmoothDiff(model, input, n::Int, std::Real, rng = GLOBAL_RNG, show_progress = true)
    T = eltype(input)
    distribution = Normal(zero(T), convert(T, std))
    return SmoothDiff(model, input, n, distribution, rng, show_progress)
end

function call_analyzer(
        input, method::SmoothDiff, output_selector::AbstractOutputSelector; kwargs...
    )
    output, vejp_fn = prepare_vejp(input, method::SmoothDiff)
    output_selection = output_selector(output)

    # Evaluate VeJP
    v = zero(output)
    v[output_selection] .= 1
    val = only(vejp_fn(v))

    return Explanation(
        val, input, output, output_selection, :SmoothDiff, :sensitivity, nothing
    )
end

function prepare_vejp(input, method::SmoothDiff)
    model = method.model
    noisy_input = similar(input)
    reset_counts!(model)
    samplingmode!(model, true)

    p = Progress(method.n; desc = "Sampling SmoothDiff...", showspeed = method.show_progress)
    for _ in 1:(method.n - 1)
        noisy_input = rand!(method.rng, method.distribution, noisy_input)
        noisy_input .+= input
        model(noisy_input) # update counts by running inference
        next!(p)
    end

    # On last step, create VeJP function
    noisy_input = rand!(method.rng, method.distribution, noisy_input)
    noisy_input .+= input
    output, vejp_fn = pullback(model, noisy_input) # create expected Jacobian operator (aka "VeJP function")
    samplingmode!(model, false)
    next!(p)
    return output, vejp_fn
end

end # module
