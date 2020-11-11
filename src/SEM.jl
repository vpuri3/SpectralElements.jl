#
module SEM

using FastGaussQuadrature
using LinearAlgebra
using SparseArrays
using Zygote
using Plots

export linspace

linspace(zi::Number,ze::Number,n::Integer) = range(zi,stop=ze,length=n)
sum(A::AbstractArray, n::Int) = Base.sum(A, dims=n)
sum(A) = Base.sum(A)

# SEM building blocks
include("interpMat.jl")
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
include("gordonHall.jl")
include("cg.jl")

include("adjoint.jl")

end
