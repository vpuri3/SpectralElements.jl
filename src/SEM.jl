#
module SEM

using FastGaussQuadrature
using LinearAlgebra
using SparseArrays

linspace(zi::Number,ze::Number,n::Integer) = range(zi,stop=ze,length=n)

# import ...

include("interpMat.jl")
include("derivMat.jl")
include("semmesh.jl")
include("semq.jl")

include("ABu.jl")
include("jac.jl")
include("grad.jl")
include("mask.jl")
include("gatherScatter.jl")
include("mass.jl")

include("ndgrid.jl")

end
