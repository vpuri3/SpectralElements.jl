#
module SpectralElements

using Reexport

@reexport using LinearAlgebra
@reexport using LinearSolve
@reexport using UnPack: @unpack
@reexport using Setfield: @set!
@reexport using Plots

using SparseArrays
using NNlib

include("Spectral/Spectral.jl")

import FastGaussQuadrature
import Zygote
#--------------------------------------#
# conveniences
Base.:*(op::Function, x::AbstractArray) = op(x)
linspace(zi::Number,ze::Number,n::Integer) = Array(range(zi,stop=ze,length=n))
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
