using SmoothedDifferentiation
using Test
using Aqua
using JET

@testset "SmoothedDifferentiation.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(SmoothedDifferentiation)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(SmoothedDifferentiation; target_defined_modules = true)
    end

    @testset "Flux" begin
        @testset "VGG preparation tests" begin
            include("test_preparation_flux.jl")
        end
        @testset "GPU tests" begin
            include("test_gpu_flux.jl")
        end
    end
end
