# Instantiate environment
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using DrWatson

using SmoothedDifferentiation
using ExplainableAI
using Flux, Metalhead
using CUDA, cuDNN
using Distributions: Normal

using Chairmarks
using JLD2

## Prepare file saving
!isdir(datadir("benchmarks")) && mkdir(datadir("benchmarks"))

function benchmark_and_save(analyzer, input, methodname, filename; n = 1, std = 0.0f0)
    path = datadir("benchmarks", filename * ".jld2")
    if isfile(path)
        @info "Results exist, skipping..." filename
        return nothing
    else
        @info "Running benchmarks..." filename
        bm = @b analyze($input, $analyzer) # run benchmarking
        @info bm.time
        jldsave(path; method = methodname, time = bm.time, n = n, std = std)
        return nothing
    end
end

function run_benchmarks(; device = cpu)
    ## Load data
    input_data = load(
        joinpath(@__DIR__, "..", "..", "heatmaps", "data", "input_batch.jld2")
    )
    input_cpu = input_data["input"]
    input = input_cpu |> device

    ## Load pretrained model
    model = VGG(19; pretrain = true).layers |> device

    ## Run analyzers
    std = 0.5f0
    distr = Normal(0.0f0, std)
    ns = union(
        (1:10)...,
        (5:5:50)...,
        (2 .^ (0:11))..., # up to 2028 samples
    )
    for n in ns
        analyzer = SmoothDiff(model, input, n, distr)
        methodname = "SmoothDiff"
        filename = "benchmark_SmoothDiff_n=$(n)_std=$(std)"
        benchmark_and_save(analyzer, input, methodname, filename; n = n, std = std)
    end
    for n in ns
        distr = Normal(0.0f0, std)
        analyzer = SmoothGrad(model, n, distr)
        methodname = "SmoothGrad"
        filename = "benchmark_SmoothGrad_n=$(n)_std=$(std)"
        benchmark_and_save(analyzer, input, methodname, filename; n = n, std = std)
    end
    return nothing
end

run_benchmarks(; device = gpu)
