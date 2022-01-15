#
module Spectral

using LinearAlgebra
import FastGaussQuadrature

import Base: summary, show                                  # printing
import Base: similar, copy, copy!                           # allocation
import Base: length, size, getindex, setindex!, IndexStyle  # indexing
import Base.Broadcast: BroadcastStyle                       # broadcast
import Base: +, -, *, /, \, adjoint                         # math
import LinearAlgebra: mul!, ldiv!, dot, norm#, axpy!, axpby!, diagonal, Diagonal

abstract type AbstractSpectralField{T,N} <: AbstractVector{T} end
abstract type AbstractSpectralOperator{T,N} end
abstract type AbstractSpectralSpace{T,N} end

include("TensorProdField.jl")
include("TensorProdOperator.jl")
include("TensorProdSpace.jl")

end #module
