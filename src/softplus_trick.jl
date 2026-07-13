## Custom layer types
struct ReluSoftplusPullback
    β::Float32
end

# Forward passes
function (r::ReluSoftplusPullback)(x)
    return relu.(x)
end

# register all parameters as non-trainable
Flux.@layer ReluSoftplusPullback trainable = ()

# Custom VJPs #

function rrule(r::ReluSoftplusPullback, x)
    y = r(x)
    function softplus_pullback(ȳ)
        J = σ(r.β * x)
        x̄ = J .* ȳ
        return (NoTangent(), x̄)
    end
    return y, softplus_pullback
end

## Custom model preparation

function add_softplus_pullback(model::Flux.Chain, β::Float32)
    layers = []
    for l in model.layers
        push!(layers, add_softplus_pullback(l, β))
    end
    return Flux.Chain(layers...)
end

function add_softplus_pullback(p::Flux.Parallel, β::Float32)
    return Flux.Parallel(p.connection, add_softplus_pullback.(p.layers, β))
end

function add_softplus_pullback(l::Union{Flux.Dense, Flux.Conv}, β)
    if l.σ == identity
        return l
    elseif l.σ != relu
        error("Unsupported activation function $(l.σ), expected relu")
    end
    layer = linear_copy(l)
    activation = ReluSoftplusPullback(β)
    return Flux.Chain(layer, activation)
end

add_softplus_pullback(layer, β) = layer # skip all other layer types

## XAIBase interface

struct SoftPlusTrick{M} <: AbstractXAIMethod
    model::M
    beta::Float32

    function SoftPlusTrick(model, β::Float32 = 1.0f0)
        prepared_model = add_softplus_pullback(model, β)
        Flux.testmode!(prepared_model, true)
        return new{typeof(prepared_model)}(prepared_model, β)
    end
end

function call_analyzer(
        input, method::SoftPlusTrick, output_selector::AbstractOutputSelector; kwargs...
    )
    model = method.model
    output = model(input)
    output_selection = output_selector(output)

    v = zero(output)
    v[output_selection] .= 1
    _, back = pullback(model, input) # create Jacobian operator (aka "VJP function")
    val = only(back(v)) # evaluate VJP

    return Explanation(
        val, input, output, output_selection, :SoftPlus, :sensitivity, nothing
    )
end
