#
module Spectral

using Reexport

@reexport using LinearAlgebra
@reexport using LinearSolve
@reexport using UnPack: @unpack
@reexport using Setfield: @set!
@reexport using SciMLBase

import SparseArrays.sparse
import FastGaussQuadrature: gausslobatto, gausslegendre, gausschebyshev
import Base.ReshapedArray
import SciMLBase: AbstractDiffEqOperator

#using RecursiveArrayTools

# overload
import Base: summary, show                          # printing
import Base: similar                                # allocation
import Base: size, getindex, setindex!, IndexStyle  # indexing
import Base.Broadcast: BroadcastStyle               # broadcast
import Base: +, -, *, /, \, adjoint, ∘              # maths
import LinearAlgebra: mul!, ldiv!, lmul!, rmul!
import Base: kron

""" Scalar function field in D-Dimensional space """
abstract type AbstractField{T,D} <: AbstractVector{T} end
""" Operators acting on fields in D-Dimensional space """
abstract type AbstractOperator{T,D} <: AbstractDiffEqOperator{T} end
""" Function space in D-Dimensional space """
abstract type AbstractSpace{T,D} end

""" Scalar function field in D-Dimensional space over spectral basis """
abstract type AbstractSpectralField{T,D} <: AbstractField{T,D} end
""" Operators acting on fields in D-Dimensional space over a spectral basis"""
abstract type AbstractSpectralOperator{T,D} <: AbstractOperator{T,D} end
""" Spectral function space in D-Dimensional space """
abstract type AbstractSpectralSpace{T,D} <: AbstractSpace{T,D} end

""" Tensor product operator in D-Dimensional space """
abstract type AbstractTensorProductOperator{T,D} <: AbstractOperator{T,D} end

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
#include("Space.jl")

export 
       # fields
       Field,

       # operator conveniences
       IdentityOp, ZeroOp, AffineOp, ComposeOp, InverseOp, # overload op(u,p,t)

       # Concrete operators
       DiagonalOp, TensorProductOp2D, # define op(u,p,t)

       # spaces
       GaussLobattoLegendre2D, GaussLegendre2D, GaussChebychev2D

end # module
