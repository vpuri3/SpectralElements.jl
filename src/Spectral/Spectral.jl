#
module Spectral

using Reexport

@reexport using LinearAlgebra
@reexport using LinearSolve
@reexport using UnPack: @unpack
@reexport using Setfield: @set!
@reexport using SciMLBase

import Base.ReshapedArray
import SciMLBase: AbstractDiffEqOperator
import Lazy: @forward

import SparseArrays: sparse
import FastGaussQuadrature: gausslobatto, gausslegendre, gausschebyshev
import FFTW: plan_rfft, plan_irfft

# Field <: AbstractVector
import Base: summary, show, similar
import Base: size, getindex, setindex!, IndexStyle
import Base.Broadcast: BroadcastStyle

# overload maths
import Base: +, -, *, /, \, adjoint, ∘, inv, kron
import LinearAlgebra: mul!, ldiv!, lmul!, rmul!

""" Scalar function field in D-Dimensional space """
abstract type AbstractField{T,D} <: AbstractVector{T} end
""" Operators acting on fields in D-Dimensional space """
abstract type AbstractOperator{T,D} <: AbstractDiffEqOperator{T} end
""" D-Dimensional physical domain """
abstract type AbstractDomain{T,D} end
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

Base.eltype(::Union{
                    AbstractField{T,D},
                    AbstractOperator{T,D},
                    AbstractSpace{T,D},
                    AbstractDomain{T,D},
                   }
           ) where{T,D} = T

dims(::Union{AbstractField{T,D},
             AbstractOperator{T,D},
             AbstractSpace{T,D},
             AbstractDomain{T,D}
             }
    ) where{T,D} = D

include("utils.jl")
include("Field.jl")
include("OperatorBasics.jl")
include("Operators.jl")
#imclude("Domain.jl")
#include("Space.jl")

export 
       # fields
       Field,

       # operator conveniences
       IdentityOp, ZeroOp, AffineOp, ComposeOp, InverseOp, # overload op(u,p,t)

       # Concrete operators
       MatrixOp, DiagonalOp, TensorProductOp2D,

       # Domains
       Interval, BoxDomain,

       # spaces
       GaussLobattoLegendre2D, GaussLegendre2D, GaussChebychev2D

end # module
