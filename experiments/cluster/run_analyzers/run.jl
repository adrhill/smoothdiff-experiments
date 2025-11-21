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

    ## SoftPlusTrick heuristic
    VARIANCE_VGG = 0.5f0
    BETA_VGG = Float32(
        log(2) *
            sqrt(2 * pi / VARIANCE_VGG) *
            sum(count(x -> isa(x, Union{Dense, Conv}), ls) for ls in model),
    )

    ## Run analyzers
    ns = union(
        (1:10)...,
        (5:5:50)...,
        (2 .^ (0:11))..., # up to 2028 samples
    )
    stds = 0.4:0.1:0.6
    for n in ns
        for std in stds
            distr = Normal(0.0f0, std)
            analyzer = SmoothDiff(model, input, n, distr)
            methodname = "SmoothDiff"
            filename = "SmoothDiff_n=$(n)_std=$(std)"
            analyze_and_save(analyzer, input, methodname, filename; n = n, std = std)
        end
    end
    for n in ns
        for std in stds
            distr = Normal(0.0f0, std)
            analyzer = SmoothGrad(model, n, distr)
            methodname = "SmoothGrad"
            filename = "SmoothGrad_n=$(n)_std=$(std)"
            analyze_and_save(analyzer, input, methodname, filename; n = n, std = std)
        end
    end
    for beta in Float32.(union(0.5, 1, 2, 3, (5:5:50)...))
        analyzer = SoftPlusTrick(model, beta)
        methodname = "SoftPlusTrick"
        filename = "SoftPlusTrick_beta=$(beta)"
        analyze_and_save(analyzer, input, methodname, filename)
    end

    analyzer = Gradient(model)
    methodname = "Gradient"
    filename = "Gradient"
    analyze_and_save(analyzer, input, methodname, filename)

    analyze_and_save(
        InputTimesGradient(model), input, "InputTimesGradient", "InputTimesGradient"
    )

    for (composite, name) in (
            (EpsilonPlus(), "LRP_EpsilonPlus"),
            (EpsilonPlusFlat(), "LRP_EpsilonPlusFlat"),
            (EpsilonAlpha2Beta1(), "LRP_EpsilonAlpha2Beta1"),
            (EpsilonAlpha2Beta1Flat(), "LRP_EpsilonAlpha2Beta1Flat"),
        )
        analyzer = LRP(model, composite)
        analyze_and_save(analyzer, input, name, name)
    end
    return nothing
end

# run_analyzers(; device=cpu)
run_analyzers(; device = gpu)
