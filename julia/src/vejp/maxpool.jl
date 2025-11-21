# Custom layer type
Base.@kwdef mutable struct MaxPoolAccumulator{L <: Flux.MaxPool, T <: AbstractArray{Int}}
    layer::L
    count::T # stateful count of positive activations
    n::Int = 0   # total number of calls
    sampling::Bool = true # indicator whether sampling of forward-passes is active
end

reset_counts!(m::MaxPoolAccumulator) = fill!(m.count, 0)
samplingmode!(m::MaxPoolAccumulator, mode::Bool) = (m.sampling = mode; m)

# Forward pass
function (m::MaxPoolAccumulator)(x)
    pool = m.layer
    pdims = Flux.PoolDims(x, pool.k; padding = pool.pad, stride = pool.stride)
    y = maxpool(x, pdims)
    if m.sampling
        dy = fill!(similar(y), 1)
        dx = ∇maxpool(dy, y, x, pdims)
        m.count .+= Bool.(dx)
        m.n += 1
    end
    return y
end

# Register all parameters as non-trainable
Flux.@layer MaxPoolAccumulator trainable = ()

# Custom VJP computing VeJP
function rrule(m::MaxPoolAccumulator, x)
    y = m(x)
    function modified_maxpool_pullback(ȳ)
        ȳ_expanded = upsample_nearest(ȳ, m.layer.stride)
        J = convert.(Float32, m.count) / m.n
        x̄ = J .* ȳ_expanded
        return (NoTangent(), x̄)
    end
    return y, modified_maxpool_pullback
end
