#
module SEM

using FastGaussQuadrature
using LinearAlgebra
using SparseArrays
using Zygote

linspace(zi::Number,ze::Number,n::Integer) = range(zi,stop=ze,length=n)

# import ...

include("interpMat.jl")
include("derivMat.jl")
include("semmesh.jl")
include("semq.jl")

include("ndgrid.jl")

include("ABu.jl")
include("jac.jl")
include("grad.jl")
include("gatherScatter.jl")
include("mask.jl")
include("mass.jl")
include("lapl.jl")

include("cg.jl")

include("adjoint.jl")

end
