#
module Spectral

using LinearAlgebra
import FastGaussQuadrature
import UnPack.@unpack
import Setfield.@set!
import Base.ReshapedArray

# overload
import Base: summary, show                          # printing
import Base: similar                                # allocation
import Base: size, getindex, setindex!, IndexStyle  # indexing
import Base.Broadcast: BroadcastStyle               # broadcast
import Base: +, -, *, /, \, adjoint, ∘              # math
import LinearAlgebra: mul!, ldiv!

abstract type AbstractSpectralField{T,N} <: AbstractVector{T} end
abstract type AbstractSpectralOperator{T,N} end
abstract type AbstractSpectralSpace{T,N} end

Base.eltype(u::Union{AbstractSpectralField{T,N},
                     AbstractSpectralOperator{T,N},
                     AbstractSpectralSpace{T,N}}
           ) where{T,N} = T

""" utilize Base.ReshaedArray """
_reshape(a,dims::NTuple{N,Int}) where{N} = reshape(a,dims) # fallback
_reshape(a::Array, dims::NTuple{N,Int}) where{N} = Base.ReshapedArray(a, dims, ())
#
_vec(a::AbstractVector) = a
_vec(a::AbstractArray) = _reshape(a,(length(a),))

""" check if operator(s) is square """
issquare(::UniformScaling) = true
issquare(A::AbstractMatrix) = size(A,1) === size(A,2)
issquare(A...) = @. (&)(issquare(A)...)

""" Identity Matrix with the notion of size """
struct Identity{Ti,Tn} #<: AbstractMatrix{Bool}
  I::Ti
  n::Tn
  #
  function Identity(n)
    id = I
    new{typeof(id),typeof(n)}(id,n)
  end
end
Base.size(Id::Identity) = (Id.n, Id.n)
Base.eltype(::Identity) = Bool
Base.adjoint(Id::Identity) = Id
#
(::Identity)(u) = u
(*)(::Identity, u) = copy(u) # unnecessary if AbstractMatrix
LinearAlgebra.mul!(v, id::Identity, u) = mul!(v, id.I, u)
LinearAlgebra.ldiv!(v, id::Identity, u) = ldiv!(v, id.I, u)
LinearAlgebra.ldiv!(id::Identity, u) = ldiv!(id.I, u)

# includes
include("TensorProdField.jl")
include("OperatorBasics.jl")
include("TensorProdOperator.jl")
include("DerivMat.jl")
include("InterpMat.jl")
#include("TensorProdSpace.jl")

export Field,
       Identity, CopyingOp, ComposeOperator, InverseOperator,
       DiagonalOp, TensorProd2DOp,
       SpectralSpace2D, GaussLobattoLegendre2D, GaussLegendre2D, GaussChebychev2D

end # module
