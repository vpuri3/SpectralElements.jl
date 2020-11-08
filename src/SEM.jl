#
module SEM

using FastGaussQuadrature
using LinearAlgebra
using SparseArrays
using Zygote
using Plots

linspace(zi::Number,ze::Number,n::Integer) = range(zi,stop=ze,length=n)

# import ...

# SEM building blocks
include("interpMat.jl")
include("derivMat.jl")
include("semmesh.jl")
include("semq.jl")

# utilities
include("ndgrid.jl")
include("plt.jl")

# spectral element
include("ABu.jl")
include("jac.jl")
include("grad.jl")
include("gatherScatter.jl")
include("mask.jl")
include("mass.jl")
include("lapl.jl")

include("cg.jl")

include("adjoint.jl")
include("gordonHall.jl")

end
