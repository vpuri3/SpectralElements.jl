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

import SciMLBase: AbstractDiffEqLinearOperator

#using RecursiveArrayTools

# overload
import Base: summary, show                          # printing
import Base: similar                                # allocation
import Base: size, getindex, setindex!, IndexStyle  # indexing
import Base.Broadcast: BroadcastStyle               # broadcast
import Base: +, -, *, /, \, adjoint, ∘              # math
import LinearAlgebra: mul!, ldiv!, lmul!, rmul!
import Base: kron

""" Abstract Scalar Function Field  in D-Dimensinoal Space"""
abstract type AbstractSciMLField{T,D} <: AbstractVector{T} end

abstract type AbstractField{T,D} <: AbstractSciMLField{T,D} end
abstract type AbstractOperator{T,D} <: AbstractDiffEqLinearOperator{T} end
abstract type AbstractSpace{T,D} end

abstract type AbstractTensorProdOperator{T,D} <: AbstractOperator{T,D} end

#""" Abstract Tensor Product Polynomial Field in 2D """
#abstract type AbstractTensorProdPoly2DField{T} <: AbstractField{T,2}
#abstract type AbstractTensorProdPoly2DOperator{T} <: AbstractOperator{T,2}
#abstract type AbstractTensorProdPoly2DSpace{T} <: AbstractOperator{T,2}

Base.eltype(::Union{AbstractField{T,D},
                    AbstractOperator{T,D},
                    AbstractSpace{T,D}}
           ) where{T,D} = T

dims(::Union{AbstractField{T,D},
             AbstractOperator{T,D},
             AbstractSpace{T,D}
             }
    ) where{T,D} = D

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
