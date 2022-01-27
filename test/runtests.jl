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
    using LinearAlgebra

    nr = 10
    ns = 12

    # Field
    u = rand(nr,ns) |> Field
    v = u .+ u

    # DiagonalOp
    d = rand(nr,ns) |> Field
    D = DiagonalOp(d)

    mul!(v,D,u)
    @test v ≈ d .* u

    # TensorProd2DOp
    Ar = rand(nr,nr)
    Bs = rand(ns,ns)
    T = TensorProd2DOp(Ar,Bs)

    mul!(v,T,u)
    @test v ≈ Field(Ar * u.array * Bs')

#   @time @safetestset "Examples" begin include("examples.jl") end
end
