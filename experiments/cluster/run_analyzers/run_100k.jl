# Instantiate environment
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using DrWatson

using SmoothedDifferentiation
using ExplainableAI
using RelevancePropagation
using Flux, Metalhead
using CUDA, cuDNN
using Distributions: Normal

using JLD2

## Hotfix for Metalhead compatibility
import Flux: loadmodel!
loadmodel!(dst::MaxPool{N, M}, src::Tuple{}; kw...) where {N, M} = dst

## Prepare file saving
!isdir(datadir()) && mkdir(datadir())

function analyze_and_save(analyzer, input, methodname, filename; n = 1, std = 0.0f0)
    path = datadir(filename * ".jld2")
    if isfile(path)
        @info "Results exist, skipping..." filename
        return nothing
    else
        @info "Running..." filename
        expl = analyze(input, analyzer)
        val = Array(expl.val) # convert to CPU array for saving
        jldsave(path; method = methodname, val = val, n = n, std = std)
        return nothing
    end
end

function run_analyzers(; device = cpu)
    ## Load data
    input_data = load(
        joinpath(@__DIR__, "..", "..", "heatmaps", "data", "input_batch.jld2")
    )
    input_cpu = input_data["input"]
    input = input_cpu |> device

    ## Load pretrained model
    model = VGG(19; pretrain = true).layers |> device

    n = 100_000
    std = 0.5f0
    distr = Normal(0.0f0, std)
    analyzer = SmoothGrad(model, n, distr)
    methodname = "SmoothGrad"
    filename = "SmoothGrad_n=$(n)_std=$(std)"
    analyze_and_save(analyzer, input, methodname, filename; n = n, std = std)
    return nothing
end

run_analyzers(; device = gpu)
