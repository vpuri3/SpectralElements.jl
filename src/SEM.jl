#
module SEM

using LinearAlgebra,SparseArrays
using Zygote,Plots
using UnPack
using StaticArrays

import FastGaussQuadrature
import Zygote,NNlib
#--------------------------------------#
Base.:*(op::Function,x::Array) = op(x)
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
