#
module SEM

using FastGaussQuadrature
using LinearAlgebra
using SparseArrays
using Zygote
using Plots

#--------------------------------------#
export linspace,iscallable

#--------------------------------------#
import Base.*
*(op::Function,x) = op(x)
#--------------------------------------#
iscallable(op) = !isempty(methods(op))
linspace(zi::Number,ze::Number,n::Integer) = Array(range(zi,stop=ze,length=n))
sum(A::AbstractArray, n::Int) = Base.sum(A, dims=n)
sum(A) = Base.sum(A)
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
include("gatherScatter.jl")
include("geom.jl")

include("pcg.jl")

include("adjoint.jl")

include("fem.jl")

end # module
