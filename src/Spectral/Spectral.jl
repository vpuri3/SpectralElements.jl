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

""" Abstract Function Field representing a scalar function in N dimensions with type T"""
abstract type AbstractField{T,N} <: AbstractVector{T} end
abstract type AbstractOperator{T,N} end
abstract type AbstractSpace{T,N} end

#""" Abstract Tensor Product Polynomial Field in 2D """
#abstract type AbstractTensorProdPoly2DField{T} <: AbstractField{T,Val{2}}
#abstract type AbstractTensorProdPoly2DOperator{T} <: AbstractOperator{T,Val{2}}
#abstract type AbstractTensorProdPoly2DSpace{T} <: AbstractOperator{T,Val{2}}

Base.eltype(u::Union{AbstractField{T,N},
                     AbstractOperator{T,N},
                     AbstractSpace{T,N}}
           ) where{T,N} = T

dims(u::Union{AbstractField{T,N},
              AbstractOperator{T,N},
              AbstractSpace{T,N}}
    ) where{T,N} = N

include("misc_utils.jl")
include("Field.jl")
include("OperatorBasics.jl")
include("Operators.jl")
#include("TensorProdSpace.jl")

export Field,
       Identity, ToArrayOp, ComposeOperator, InverseOperator,
       DiagonalOp, TensorProd2DOp,
       TensorProductSpace2D, GaussLobattoLegendre2D, GaussLegendre2D, GaussChebychev2D

end # module
