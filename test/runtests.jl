using SpectralElements
using Test, SafeTestsets
import Pkg

function activate_env(dir)
    Pkg.activate(dir)
    Pkg.develop(Pkg.PackageSpec(path=dirname(@__DIR__)))
    Pkg.instantiate()
end

@testset "SpectralElements.jl" begin

    using SpectralElements.Spectral
    u = Spectral.Field(rand(10,10))
    v = u'
    D = Spectral.DiagonalOp(u)
    v = u .+ u

    @time @safetestset "Examples" begin include("examples.jl") end
end
