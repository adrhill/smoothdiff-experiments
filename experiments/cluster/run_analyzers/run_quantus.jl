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
import XAIBase: AbstractOutputSelector

using HDF5

## Hotfix for Metalhead compatibility
import Flux: loadmodel!
loadmodel!(dst::MaxPool{N, M}, src::Tuple{}; kw...) where {N, M} = dst

## Input and output directories
const input_dir = joinpath(@__DIR__, "quantus_samples")
const results_dir = datadir("quantus")
!isdir(results_dir) && mkdir(results_dir)

struct BatchSelector{I} <: AbstractOutputSelector
    index::I
end
function (s::BatchSelector)(out::AbstractMatrix)
    length(s.index) != size(out, 2) && throw(DimensionMismatch("Mismatch in batch dimension of output tensor and indices of output selector."))
    return [CartesianIndex{2}(idx, batchdim) for (batchdim, idx) in enumerate(s.index)]
end

function analyze_and_save(analyzer, input, output_selection, methodname, filename; n = 1, std = 0.0f0)
    path = joinpath(results_dir, filename * ".h5")
    if isfile(path)
        @info "Results exist, skipping..." filename
        return nothing
    else
        @info "Running..." filename
        expl = analyze(input, analyzer, output_selection)

        a = Array(expl.val) # convert to CPU array for saving
        h5open(path, "w") do fid
            fid["a"] = a # Save in WHCN format
        end
        return nothing
    end
end

function run_analyzers(; device = cpu)
    ## Load data
    input = h5open(joinpath(input_dir, "x_batch.h5"), "r") do file
        convert.(Float32, read(file, "x")) |> device
    end
    targets = h5open(joinpath(input_dir, "y_batch.h5"), "r") do file
        read(file, "y") .+ 1
    end
    output_selection = BatchSelector(targets)

    ## Load pretrained model
    model = VGG(19; pretrain = true).layers |> device

    ## Run analyzers
    std = 0.5
    distr = Normal(0.0f0, std)

    begin
        analyze_and_save(
            Gradient(model), input, output_selection,
            "Gradient", "Gradient"
        )
    end
    begin
        analyze_and_save(
            InputTimesGradient(model), input, output_selection,
            "InputTimesGradient", "InputTimesGradient"
        )
    end

    for n in (10, 50)
        begin
            analyzer = SmoothDiff(model, input, n, distr)
            analyze_and_save(
                analyzer, input, output_selection,
                "SmoothDiff", "SmoothDiff_n=$(n)_std=$(std)"
            )
        end
        begin
            analyzer = SmoothGrad(model, n, distr)
            analyze_and_save(
                analyzer, input, output_selection,
                "SmoothGrad", "SmoothGrad_n=$(n)_std=$(std)"
            )
        end
        begin
            analyze_and_save(
                IntegratedGradients(model, n), input, output_selection,
                "IntegratedGradients", "IntegratedGradients",
            )
        end
    end

    # BetaSmoothing
    for beta in Float32.((0.5, 1, 2))
        analyzer = SoftPlusTrick(model, beta)
        analyze_and_save(
            analyzer, input, output_selection,
            "BetaSmoothing", "BetaSmoothing_beta=$(beta)"
        )
    end

    # LRP
    for (composite, name) in (
            (EpsilonPlus(), "LRP_EpsilonPlus"),
            (EpsilonPlusFlat(), "LRP_EpsilonPlusFlat"),
            (EpsilonAlpha2Beta1(), "LRP_EpsilonAlpha2Beta1"),
            (EpsilonAlpha2Beta1Flat(), "LRP_EpsilonAlpha2Beta1Flat"),
            (Composite(ZeroRule()), "LRP_Zero"),
        )
        analyzer = LRP(model, composite)
        analyze_and_save(analyzer, input, output_selection, name, name)
    end
    return nothing
end

run_analyzers(; device = cpu)
# run_analyzers(; device = gpu)
