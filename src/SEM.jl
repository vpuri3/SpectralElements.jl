#
module SEM

using LinearAlgebra,SparseArrays
using Zygote,Plots
using UnPack
using StaticArrays

import FastGaussQuadrature
#--------------------------------------#
Base.:*(op::Function,x::AbstractArray) = op(x)
#--------------------------------------#
export linspace
linspace(zi::Number,ze::Number,n::Integer) = Array(range(zi,stop=ze,length=n))
#--------------------------------------#
export iscallable
iscallable(op) = !isempty(methods(op))
#--------------------------------------#
sum(A::AbstractArray, n::Int) = Base.sum(A, dims=n)
sum(A) = Base.sum(A)
#cumprod(A::AbstractArray) = Base.cumprod(A, dims=1)
#cumprod(A::AbstractArray, d::Int) = Base.cumprod(A, dims=d)
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

include("mesh.jl")

include("pcg.jl")
include("adjoint.jl")
include("fem.jl")

include("gatherScatter.jl")
include("grad.jl")
include("mass.jl")
include("lapl.jl")
include("hlmz.jl")

include("plt.jl")

end # module
