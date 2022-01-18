#
module Spectral

using LinearAlgebra
import FastGaussQuadrature

import Base: summary, show                                  # printing
import Base: similar, copy, copy!                           # allocation
import Base: length, size, getindex, setindex!, IndexStyle  # indexing
import Base.Broadcast: BroadcastStyle                       # broadcast
import Base: +, -, *, /, \, adjoint                         # math
import LinearAlgebra: mul!, ldiv!, dot, norm, Adjoint#, Diagonal

abstract type AbstractSpectralField{T,N} <: AbstractVector{T} end
abstract type AbstractSpectralOperator{T,N} end
abstract type AbstractSpectralSpace{T,N} end

eltype(u::AbstractSpectralField{T,N}) where{T,N} = T
eltype(u::AbstractSpectralOperator{T,N}) where{T,N} = T
eltype(u::AbstractSpectralSpace{T,N}) where{T,N} = T

function (A::AbstractSpectralOperator{T,N})(u)
    v = similar(u)
    mul!(v,A,u)
end

include("TensorProdField.jl")
include("TensorProdOperator.jl")
include("TensorProdSpace.jl")

end #module
