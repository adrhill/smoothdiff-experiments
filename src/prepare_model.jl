## Reset counters
reset_counts!(layer) = layer

function reset_counts!(c::Flux.Chain)
    for l in c.layers
        reset_counts!(l)
    end
    return nothing
end

function reset_counts!(p::Flux.Parallel)
    for l in p.layers
        reset_counts!(l)
    end
    return nothing
end

## Unpack and iterate over wrappers and containers
function prepare(model::Flux.Chain, x)
    layers = []
    for l_in in model.layers
        l_out = prepare(l_in, x)
        push!(layers, l_out)
        x = l_in(x)
    end
    return Flux.Chain(layers...)
end

prepare(p::Flux.Parallel, x) = Flux.Parallel(
    prepare(p.connection, x),
    prepare.(p.layers, Ref(x))
)

## Replace nonlinear layers in model
prepare(layer, x) = layer # by default, don't replace layer
prepare(l::typeof(relu), x) = ReluAccumulator(count = similar(x, Int))
prepare(l::Flux.MaxPool, x) = MaxPoolAccumulator(layer = l, count = similar(x, Int))

function prepare(l::Union{Flux.Dense, Flux.Conv}, x)
    y = l(x)
    if l.σ == identity
        return l
    elseif l.σ != relu
        error("Unsupported activation function $(l.σ), expected relu")
    end

    layer = linear_copy(l)
    activation = ReluAccumulator(count = similar(y, Int))
    return Flux.Chain(layer, activation)
end

## Copy layers with linear activation functions
linear_copy(l::Flux.Dense) = Flux.Dense(l.weight, l.bias, identity)
linear_copy(l::Flux.Conv) = Flux.Conv(
    l.weight,
    l.bias,
    identity;
    stride = l.stride,
    pad = l.pad,
    dilation = l.dilation,
    groups = l.groups,
)
