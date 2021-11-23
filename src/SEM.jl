#
module SEM

using Reexport
@reexport using DiffEqBase

using LinearAlgebra,SparseArrays
using Zygote,Plots
using UnPack
using StaticArrays

import FastGaussQuadrature
import Zygote,NNlib
#--------------------------------------#
Base.:*(op::Function, x::AbstractArray) = op(x)
#--------------------------------------#
export linspace
linspace(zi::Number,ze::Number,n::Integer) = Array(range(zi,stop=ze,length=n))
#--------------------------------------#
export iscallable
iscallable(op) = !isempty(methods(op))
#--------------------------------------#
sum(A::Array, n::Int) = Base.sum(A, dims=n)
sum(A) = Base.sum(A)
#cumprod(A::Array) = Base.cumprod(A, dims=1)
#cumprod(A::Array, d::Int) = Base.cumprod(A, dims=d)
#flipdim(A, d) = reverse(A, dims=d)
#--------------------------------------#

include("Spectral/Spectral.jl")

#=
abstract type AbstractSpectralElementMesh end
abstract type AbstractSpectralElementOperator{T} <: AbstractMatrix{T} end
abstract type AbstractSpectralElementField{T}    <: AbstractVector{T} end

abstract type AbstractGatherScatter end

mutable struct GatherScatter{T,N}
    QQtx::AbstractArray{T,}
    QQty::AbstractArray{T,}
    mult::AbstractArray{T,N}
end

mutable struct SpectralElementMesh{T,Ti,N}
    E::Vector{Ti}
    n::Vector{Ti}

    space::Discretization{T}
    gs::GatherScatte{T,N}

    geom::Geometry
    #x::AbstractArray{T,N}
    #y::AbstractArray{T,N}
    # J, Ji, rx,ry,sx,sy, B, Bi,
end

mutable struct LaplaceOperator{T,N} <: SpectralElementOperator
    G::AbstractVector{AbstractArray{T}}
    msh::Tmsh{T,Ti,N}
end

mutable struct SpectralElementField{Tu <: AbstractArray{T,N},
#                           Tbc, Tm,
                            Tmsh{T,N}} <: AbstractSpectralElementField{T,N}
    u::Tu
#   ub::Tu
#   bc::Tbc
#   M::Tm
    msh::Tmsh
end

#Base.:*

"""
 DiffEq ecosystem bindings
"""

=#

#--------------------------------------#

# SEM building blocks
include("jac.jl")
include("interp.jl")
include("derivMat.jl")
include("semmesh.jl")
include("semq.jl")
include("ndgrid.jl")

include("ABu.jl")
include("geom.jl")
include("mask.jl")

include("time.jl")
include("mesh.jl")

include("pcg.jl")
include("adjoint.jl")

include("gatherScatter.jl")
include("grad.jl")
include("mass.jl")
include("lapl.jl")
include("hlmz.jl")
include("advect.jl")
include("diver.jl")

include("diffusion.jl")
include("convectionDiffusion.jl")
#include("stokes.jl")

include("plt.jl")
#--------------------------------------#

end # module
