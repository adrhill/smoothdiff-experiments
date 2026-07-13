# Custom layer type
Base.@kwdef mutable struct ReluAccumulator{T <: AbstractArray{Int}}
    count::T # stateful count of positive activations
    n::Int = 0   # total number of calls
    sampling::Bool = true # indicator whether sampling of forward-passes is active
end

reset_counts!(r::ReluAccumulator) = fill!(r.count, 0)
samplingmode!(m::ReluAccumulator, mode::Bool) = (m.sampling = mode; m)

function Base.show(io::IO, r::ReluAccumulator{T}) where {T}
    return print(io, "ReluAccumulator{$T}(n=$(r.n), sampling=$(r.sampling))")
end

# Forward pass
function (r::ReluAccumulator)(x)
    if r.sampling
        @. r.count += x > 0
        r.n += 1
    end
    return relu.(x)
end

# Register all parameters as non-trainable
Flux.@layer ReluAccumulator trainable = ()

# Custom VJP computing VeJP
function rrule(r::ReluAccumulator, x)
    y = r(x)
    function modified_relu_pullback(ȳ)
        J = convert.(Float32, r.count) / r.n
        x̄ = J .* ȳ
        return (NoTangent(), x̄)
    end
    return y, modified_relu_pullback
end
