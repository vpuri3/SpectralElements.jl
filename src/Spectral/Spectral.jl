#
module Spectral

using LinearAlgebra
import FastGaussQuadrature
using UnPack#, Setfield # caching

import Base: summary, show                                  # printing
import Base: similar, copy, copy!                           # allocation
import Base: length, size, getindex, setindex!, IndexStyle  # indexing
import Base.Broadcast: BroadcastStyle                       # broadcast
import Base: +, -, *, /, \, adjoint                         # math
import LinearAlgebra: mul!, ldiv!, dot, norm, Adjoint#, Diagonal

abstract type AbstractSpectralField{T,N} <: AbstractVector{T} end
abstract type AbstractSpectralOperator{T,N} end
abstract type AbstractSpectralSpace{T,N} end

eltype(u::Union{AbstractSpectralField{T,N},
                AbstractSpectralOperator{T,N},
                AbstractSpectralSpace{T,N}}
      ) where{T,N} = T

function (A::AbstractSpectralOperator)(u) 
    if issquare(A)
        mul!(similar(u),A,u)
    else
        ArgumentError("Operation not defined for $A")
    end
end

import Base.ReshapedArray
_reshape(a,dims::NTuple{N,Int}) where{N} = reshape(a,dims)
_reshape(a::Array, dims::NTuple{N,Int}) where{N} = Base.ReshapedArray(a, dims, ())

_vec(a::AbstractVector) = a
_vec(a::AbstractArray) = _reshape(a,(length(a),))

issquare(::Union{UniformScaling, DiagonalOp}) = true
issquare(A::AbstractMatrix) = size(A,1) === size(A,2)
issquare(A...) = @. (&)(issquare(A)...)

""" ComposeOperator """
struct ComposeOperator{T,N,Ti,To} <: AbstractSpectralOperator{T,N}
    inner::Ti
    outer::To
    #
    function ComposeOperator(inner::AbstractSpectralOperator{Ti,N},
                             outer::AbstractSpectralOperator{To,N}
                            ) where{Ti,To,N}
        T = promote_type(Ti, To)
        new{T,N,typeof(inner),typeof(outer)}(inner,outer)
    end
end
(A::ComposeOperator)(u) = A.outer(A.inner(u))
function Base.:∘(outer::AbstractSpectralOperator,
                 inner::AbstractSpectralOperator)
    ComposeOperator(inner,outer)
end
size(A::ComposeOperator) = (size(A.outer, 1), size(A.inner, 2))

function LinearAlgebra.ldiv!(A::ComposeOperator, x)
    @unpack inner, outer = A

    ldiv!(inner, x)
    ldiv!(outer, x)
end

function LinearAlgebra.ldiv!(y, A::ComposeOperator, x)
    @unpack inner, outer = A

    ldiv!(y, inner, x)
    ldiv!(outer, y)
end

""" InverseOperator """
struct InverseOperator{T,N,Ta} <: AbstractSpectralOperator{T,N}
    A::Ta
    #
    function InverseOperator(A::AbstractSpectralOperator{T,N}) where{T,N}
        LinearAlgebra.checksquare(A)
        new{T,N,typeof(A)}(A)
    end
end

inv(A::AbstractSpectralOperator) = InverseOperator(A)
size(A::InverseOperator) = size(A.A)
# https://github.com/SciML/LinearSolve.jl/issues/97
LinearAlgebra.ldiv!(A::InverseOperator, x) = mul!(x, A.A, x)
LinearAlgebra.ldiv!(y, A::InverseOperator, x) = mul!(y, A.A, x)
LinearAlgebra.mul!(y, A::InverseOperator, x) = ldiv!(y, A.A, x)

include("TensorProdField.jl")
include("TensorProdOperator.jl")
include("TensorProdSpace.jl")

end #module
