using SpectralElements
using Test
using Pkg, SafeTestsets

function activate_examples_env()
    Pkg.activate("../examples")
    Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
    Pkg.instantiate()
end

@testset "SpectralElements.jl" begin

    using SpectralElements.Spectral
    u = Spectral.Field(rand(10,10))
    v = u'
    D = Spectral.DiagonalOp(u)
    v = u .+ u
    @test size(v.u) == (10,10) # broadcasting works ok

    activate_examples_env()
#   @time @safetestset "Examples" begin include("examples.jl") end
end
