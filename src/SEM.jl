#
module SEM

using LinearAlgebra, SparseArrays
using Zygote
using Plots
using UnPack

import FastGaussQuadrature
import Zygote, NNlib
#--------------------------------------#
Base.:*(op::Function, x::AbstractArray) = op(x)
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
#function dolinsolve(integrator, linsolve; A = nothing, u = nothing, b = nothing,
#                             Pl = nothing, Pr = nothing,
#                             reltol = integrator === nothing ? nothing : integrator.opts.reltol)
#  A !== nothing && (linsolve = LinearSolve.set_A(linsolve,A))
#  b !== nothing && (linsolve = LinearSolve.set_b(linsolve,b))
#  u !== nothing && (linsolve = LinearSolve.set_u(linsolve,u))
#  (Pl !== nothing || Pr !== nothing) && (linsolve = LinearSolve.set_prec(Pl,Pr))
#
#  linres = if reltol === nothing
#    solve(linsolve;reltol)
#  else
#    solve(linsolve;reltol)
#  end
#end
#--------------------------------------#

end # module
