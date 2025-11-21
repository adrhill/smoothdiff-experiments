using SmoothedDifferentiation
using Test

using NNlib
using Flux: Flux
using MLUtils: flatten
using Metalhead: Metalhead
using SmoothedDifferentiation: ReluAccumulator, MaxPoolAccumulator
using Random
using XAIBase: Explanation

model = Metalhead.VGG(11)
input = rand(Float32, 224, 224, 3, 1)
analyzer = SmoothDiff(model, input, 5)

@testset "Dry-run" begin
    expl = analyze(input, analyzer)
    @test expl isa Explanation
end

@testset "Prepare model" begin
    # Test looks very odd due to super specific nested type
    @test typeof(analyzer.model) == Flux.Chain{
        Tuple{
            Flux.Chain{
                Tuple{
                    Flux.Chain{
                        Tuple{
                            Flux.Conv{2, 2, typeof(identity), Array{Float32, 4}, Vector{Float32}},
                            ReluAccumulator{Array{Int64, 4}},
                        },
                    },
                    MaxPoolAccumulator{Flux.MaxPool{2, 4}, Array{Int64, 4}},
                    Flux.Chain{
                        Tuple{
                            Flux.Conv{2, 2, typeof(identity), Array{Float32, 4}, Vector{Float32}},
                            ReluAccumulator{Array{Int64, 4}},
                        },
                    },
                    MaxPoolAccumulator{Flux.MaxPool{2, 4}, Array{Int64, 4}},
                    Flux.Chain{
                        Tuple{
                            Flux.Conv{2, 2, typeof(identity), Array{Float32, 4}, Vector{Float32}},
                            ReluAccumulator{Array{Int64, 4}},
                        },
                    },
                    Flux.Chain{
                        Tuple{
                            Flux.Conv{2, 2, typeof(identity), Array{Float32, 4}, Vector{Float32}},
                            ReluAccumulator{Array{Int64, 4}},
                        },
                    },
                    MaxPoolAccumulator{Flux.MaxPool{2, 4}, Array{Int64, 4}},
                    Flux.Chain{
                        Tuple{
                            Flux.Conv{2, 2, typeof(identity), Array{Float32, 4}, Vector{Float32}},
                            ReluAccumulator{Array{Int64, 4}},
                        },
                    },
                    Flux.Chain{
                        Tuple{
                            Flux.Conv{2, 2, typeof(identity), Array{Float32, 4}, Vector{Float32}},
                            ReluAccumulator{Array{Int64, 4}},
                        },
                    },
                    MaxPoolAccumulator{Flux.MaxPool{2, 4}, Array{Int64, 4}},
                    Flux.Chain{
                        Tuple{
                            Flux.Conv{2, 2, typeof(identity), Array{Float32, 4}, Vector{Float32}},
                            ReluAccumulator{Array{Int64, 4}},
                        },
                    },
                    Flux.Chain{
                        Tuple{
                            Flux.Conv{2, 2, typeof(identity), Array{Float32, 4}, Vector{Float32}},
                            ReluAccumulator{Array{Int64, 4}},
                        },
                    },
                    MaxPoolAccumulator{Flux.MaxPool{2, 4}, Array{Int64, 4}},
                },
            },
            Flux.Chain{
                Tuple{
                    typeof(flatten),
                    Flux.Chain{
                        Tuple{
                            Flux.Dense{typeof(identity), Matrix{Float32}, Vector{Float32}},
                            ReluAccumulator{Matrix{Int64}},
                        },
                    },
                    Flux.Dropout{Float64, Colon, Random.TaskLocalRNG},
                    Flux.Chain{
                        Tuple{
                            Flux.Dense{typeof(identity), Matrix{Float32}, Vector{Float32}},
                            ReluAccumulator{Matrix{Int64}},
                        },
                    },
                    Flux.Dropout{Float64, Colon, Random.TaskLocalRNG},
                    Flux.Dense{typeof(identity), Matrix{Float32}, Vector{Float32}},
                },
            },
        },
    }
end
