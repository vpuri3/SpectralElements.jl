#
module Spectral

using Reexport

@reexport using LinearAlgebra
@reexport using LinearSolve
@reexport using UnPack: @unpack
@reexport using Setfield: @set!
@reexport using SciMLBase

import FastGaussQuadrature
import Base.ReshapedArray

using RecursiveArrayTools

# overload
import Base: summary, show                          # printing
import Base: similar                                # allocation
import Base: size, getindex, setindex!, IndexStyle  # indexing
import Base.Broadcast: BroadcastStyle               # broadcast
import Base: +, -, *, /, \, adjoint, ∘              # math
import LinearAlgebra: mul!, ldiv!, lmul!, rmul!

""" Abstract Scalar Function Field  in N-Dimensinoal Space"""
abstract type AbstractField{T,N} <: AbstractVector{T} end
abstract type AbstractOperator{T,N} <: AbstractDiffEqLinearOperator{T} end
abstract type AbstractSpace{T,N} end

#""" Abstract Tensor Product Polynomial Field in 2D """
#abstract type AbstractTensorProdPoly2DField{T} <: AbstractField{T,2}
#abstract type AbstractTensorProdPoly2DOperator{T} <: AbstractOperator{T,2}
#abstract type AbstractTensorProdPoly2DSpace{T} <: AbstractOperator{T,2}

Base.eltype(u::Union{AbstractField{T,N},
                     AbstractOperator{T,N},
                     AbstractSpace{T,N}}
           ) where{T,N} = T

dims(u::Union{AbstractField{T,N},
              AbstractOperator{T,N},
              AbstractSpace{T,N}}
    ) where{T,N} = N

include("utils.jl")
include("Field.jl")
include("OperatorBasics.jl")
include("Operators.jl")
#include("TensorProdSpace.jl")

export Field,
       Identity, ToArrayOp, ComposeOperator, InverseOperator,
       DiagonalOp, TensorProd2DOp,
       TensorProduct2DSpace, GaussLobattoLegendre2D, GaussLegendre2D, GaussChebychev2D

end # module
