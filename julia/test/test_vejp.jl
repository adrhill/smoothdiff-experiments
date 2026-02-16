using SmoothedDifferentiation
using SmoothedDifferentiation: ReluAccumulator, MaxPoolAccumulator, reset_counts!
using Test
using Flux: Flux, MaxPool
using NNlib: relu
using Zygote: pullback

@testset "ReLU VEJP" begin
    # 4 stats inputs, then 1 smooth backward pass.
    # Position:       1  2  3  4  5
    # Input 1:       [1, 0, 0, 0, 0]  →  count += [1, 0, 0, 0, 0]
    # Input 2:       [1, 1, 0, 0, 0]  →  count += [1, 1, 0, 0, 0]
    # Input 3:       [1, 1, 1, 0, 0]  →  count += [1, 1, 1, 0, 0]
    # Input 4:       [1, 1, 1, 1, 0]  →  count += [1, 1, 1, 1, 0]
    # Expected VEJP: count / n = [4/4, 3/4, 2/4, 1/4, 0/4]
    #              = [1.0, 0.75, 0.5, 0.25, 0.0]

    layer = ReluAccumulator(; count = zeros(Int, 5))

    stats_inputs = [
        Float32[1, 0, 0, 0, 0],
        Float32[1, 1, 0, 0, 0],
        Float32[1, 1, 1, 0, 0],
        Float32[1, 1, 1, 1, 0],
    ]
    for x in stats_inputs
        layer(x)
    end

    # Compute VeJP with grad_output = ones
    test_input = Float32[1, 1, 1, 1, 1]
    _, vejp_fn = pullback(layer, test_input)
    grad = only(vejp_fn(ones(Float32, 5)))

    @test grad ≈ Float32[1.0, 0.75, 0.5, 0.25, 0.0]
end

@testset "MaxPool VEJP" begin
    # 4 stats inputs (W=2, H=2, C=1, N=1), kernel_size=2, stride=2.
    # The max is at the top-left position in 3 of 4 inputs
    # and at the top-right in 1 of 4.
    # Expected VEJP: [0.75 0.0; 0.25 0.0] (in Julia's column-major WHCN layout)

    pool = MaxPool((2, 2); stride = (2, 2))
    count = zeros(Int, 2, 2, 1, 1)
    layer = MaxPoolAccumulator(; layer = pool, count = count)

    # In Julia WHCN layout: array[w, h, c, n]
    # "max at (w=1, h=1)" = top-left
    stats_inputs = [
        Float32[1 0; 0 0;;;],  # max at (1,1)
        Float32[1 0; 0 0;;;],  # max at (1,1)
        Float32[1 0; 0 0;;;],  # max at (1,1)
        Float32[0 0; 1 0;;;],  # max at (2,1)
    ]
    for x in stats_inputs
        layer(x)
    end

    # Compute VeJP with grad_output = ones
    test_input = Float32[1 1; 1 1;;;]
    _, vejp_fn = pullback(layer, test_input)
    grad = only(vejp_fn(ones(Float32, 1, 1, 1, 1)))

    @test grad ≈ Float32[0.75 0.0; 0.25 0.0;;;]
end
