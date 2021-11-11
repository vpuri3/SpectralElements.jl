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
"""
 DiffEq ecosystem bindings
"""
#abstract type SEMPDEAlgorithm <: AbstractPDEAlgorithm end
#
#struct EllipticPDEAlgorithm <: SEMPDEAlgorithm end
#
#function DiffEqBase.__solve{}(
#                              prob::AbstractSEMPDEProblem
#                             )
#
#    build_solution(prob,alg,ts,timeseries,
#                   du = dures,
#                   dense = dense,
#                   timeseries_errors = timeseries_errors,
#                   retcode = :Success)
#end


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

end # module
#--------------------------------------#
