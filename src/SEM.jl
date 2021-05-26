#
module SEM

using LinearAlgebra,SparseArrays
import FastGaussQuadrature
using Zygote,Plots
using UnPack

import Base:+,-,*,/,\
#--------------------------------------#
export linspace,iscallable

#--------------------------------------#
*(op::Function,x) = op(x)
#--------------------------------------#
iscallable(op) = !isempty(methods(op))
linspace(zi::Number,ze::Number,n::Integer) = Array(range(zi,stop=ze,length=n))
sum(A::AbstractArray, n::Int) = Base.sum(A, dims=n)
sum(A) = Base.sum(A)
#cumprod(A::AbstractArray) = Base.cumprod(A, dims=1)
#cumprod(A::AbstractArray, d::Int) = Base.cumprod(A, dims=d)
#flipdim(A, d) = reverse(A, dims=d)
#--------------------------------------#

# SEM building blocks
include("interp.jl")
include("derivMat.jl")
include("semmesh.jl")
include("semq.jl")

# utilities
include("ndgrid.jl")
include("plt.jl")

# operators
include("ABu.jl")
include("jac.jl")
include("grad.jl")
include("mask.jl")
include("mass.jl")
include("lapl.jl")
include("hlmz.jl")
include("gatherScatter.jl")
include("geom.jl")

include("pcg.jl")
include("adjoint.jl")
include("fem.jl")

include("mesh.jl")

end # module
