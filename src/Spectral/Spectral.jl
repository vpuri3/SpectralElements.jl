#
module Spectral

using LinearAlgebra
import FastGaussQuadrature
import UnPack.@unpack
import Setfield.@set!
import Base.ReshapedArray

# overload
import Base: summary, show                                  # printing
import Base: similar, copy, copy!                           # allocation
import Base: length, size, getindex, setindex!, IndexStyle  # indexing
import Base.Broadcast: BroadcastStyle                       # broadcast
import Base: +, -, *, /, \, adjoint                         # math
import LinearAlgebra: mul!, ldiv!, dot, norm

abstract type AbstractSpectralField{T,N} <: AbstractVector{T} end
abstract type AbstractSpectralOperator{T,N} end
abstract type AbstractSpectralSpace{T,N} end

eltype(u::Union{AbstractSpectralField{T,N},
                AbstractSpectralOperator{T,N},
                AbstractSpectralSpace{T,N}}
      ) where{T,N} = T

_reshape(a,dims::NTuple{N,Int}) where{N} = reshape(a,dims)
_reshape(a::Array, dims::NTuple{N,Int}) where{N} = Base.ReshapedArray(a, dims, ())

_vec(a::AbstractVector) = a
_vec(a::AbstractArray) = _reshape(a,(length(a),))

issquare(::UniformScaling) = true
issquare(A::AbstractMatrix) = size(A,1) === size(A,2)
issquare(A...) = @. (&)(issquare(A)...)

include("TensorProdField.jl")
include("TensorProdOperator.jl")
#include("TensorProdSpace.jl")

export Field,
       DiagonalOp, CopyingOp, TensorProductOp,
       SpectralSpace2D, GaussLobattoLegendre2D, GaussLegendre2D, GaussChebychev2D

end # module
