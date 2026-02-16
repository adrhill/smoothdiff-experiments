using SmoothedDifferentiation
using SmoothedDifferentiation: ReluAccumulator, MaxPoolAccumulator, reset_counts!
using Test
using Flux: Flux, MaxPool
using NNlib: relu
using Zygote: pullback

@testset "ReLU VEJP" begin
    # 5 inputs total: 4 stats inputs + 1 pullback call (which also counts).
    # Position:       1  2  3  4  5
    # Input 1:       [1, 0, 0, 0, 0]  →  count += [1, 0, 0, 0, 0]
    # Input 2:       [1, 1, 0, 0, 0]  →  count += [1, 1, 0, 0, 0]
    # Input 3:       [1, 1, 1, 0, 0]  →  count += [1, 1, 1, 0, 0]
    # Input 4:       [1, 1, 1, 1, 0]  →  count += [1, 1, 1, 1, 0]
    # Pullback input: [1, 1, 1, 1, 1]  →  count += [1, 1, 1, 1, 1]
    # Total count:   [5, 4, 3, 2, 1], n = 5
    # Expected VEJP: count / n = [1.0, 0.8, 0.6, 0.4, 0.2]

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

    # pullback also triggers a forward pass that increments count and n
    test_input = Float32[1, 1, 1, 1, 1]
    _, vejp_fn = pullback(layer, test_input)
    grad = only(vejp_fn(ones(Float32, 5)))

    @test grad ≈ Float32[1.0, 0.8, 0.6, 0.4, 0.2]
end

@testset "MaxPool VEJP" begin
    # 10 inputs total (WHCN layout): 9 stats inputs + 1 pullback call.
    # kernel_size=2, stride=2 on 2x2x1x1 inputs.
    # The pullback input has a clear max at (1,1) to avoid tie-breaking differences.
    # Total count: [4 2; 3 1;;;;], n = 10
    # Expected VEJP: [0.4 0.2; 0.3 0.1]

    pool = MaxPool((2, 2); stride = (2, 2))
    count = zeros(Int, 2, 2, 1, 1)
    layer = MaxPoolAccumulator(; layer = pool, count = count)

    # In Julia WHCN layout: array[w, h, c, n]
    stats_inputs = [
        Float32[1 0; 0 0;;;;],  # 3x max at (1,1)
        Float32[1 0; 0 0;;;;],
        Float32[1 0; 0 0;;;;],
        Float32[0 0; 1 0;;;;],  # 3x max at (2,1)
        Float32[0 0; 1 0;;;;],
        Float32[0 0; 1 0;;;;],
        Float32[0 1; 0 0;;;;],  # 2x max at (1,2)
        Float32[0 1; 0 0;;;;],
        Float32[0 0; 0 1;;;;],  # 1x max at (2,2)
    ]
    for x in stats_inputs
        layer(x)
    end

    # pullback also triggers a forward pass
    test_input = Float32[4 1; 1 1;;;;]
    _, vejp_fn = pullback(layer, test_input)
    grad = only(vejp_fn(ones(Float32, 1, 1, 1, 1)))

    @test grad ≈ Float32[0.4 0.2; 0.3 0.1;;;;]
end
