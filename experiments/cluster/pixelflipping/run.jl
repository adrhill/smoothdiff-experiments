# Instantiate environment
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using DrWatson
using DataFrames, JLD2
using PixelFlipper
using Flux, Metalhead

using CUDA, cuDNN

## Hotfix for Metalhead compatibility
import Flux: loadmodel!
loadmodel!(dst::MaxPool{N, M}, src::Tuple{}; kw...) where {N, M} = dst

## Prepare file saving
!isdir(datadir()) && mkdir(datadir())
!isdir(datadir("run_analyzers")) && mkdir(datadir("run_analyzers"))
!isdir(datadir("sg10k")) && mkdir(datadir("sg10k"))

function pixelflip_and_save(m::DataFrameRow, pf, model, input)
    input_file = last(splitpath(m.path))
    path = datadir("run_analyzers", "pf_$(input_file)")

    if isfile(path)
        @info "Results exist, skipping..." input_file
        return nothing
    else
        @info "Running pixel flipping on..." input_file
        res = evaluate(pf, model, input, m.val)
        MIF = mif(res)
        LIF = lif(res)
        SRG = srg(res)

        @info "SRG:" srg(res)
        display(unicode_plot(res))

        jldsave(path; method = m.method, n = m.n, std = m.std, MIF = MIF, LIF = LIF, SRG = SRG)
        return nothing
    end
end

function run_pf(; device = cpu)
    ## Load data
    input_data = load(
        joinpath(@__DIR__, "..", "..", "heatmaps", "data", "input_batch.jld2")
    )
    input_cpu = input_data["input"]
    input = input_cpu |> device

    ## Load pretrained model
    model = VGG(19; pretrain = true).layers |> device

    ## Run pixel flipping
    pf = PixelFlipping(; steps = 1000, device = device)
    df = collect_results(
        joinpath(@__DIR__, "..", "run_analyzers", "data"),
    )
    for m in eachrow(df)
        pixelflip_and_save(m, pf, model, input)
    end
    return nothing
end

run_pf(; device = gpu)
