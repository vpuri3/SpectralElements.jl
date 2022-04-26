#
module Spectral

using Reexport

@reexport using LinearAlgebra
@reexport using SciMLBase

using UnPack: @unpack
using Setfield: @set!
using LinearSolve

import Base.ReshapedArray
import SciMLBase: AbstractDiffEqOperator
import Lazy: @forward

import SparseArrays: sparse
import NNlib: gather, scatter
import FastGaussQuadrature: gausslobatto, gausslegendre, gausschebyshev
import FFTW: plan_rfft, plan_irfft

# AbstractVector subtyping
import Base: summary, show, similar, zero, one
import Base: size, getindex, setindex!, IndexStyle
import Base.Broadcast: BroadcastStyle

# overload maths
import Base: +, -, *, /, \, adjoint, ∘, inv
import LinearAlgebra: mul!, ldiv!, lmul!, rmul!

###
# Abstract Supertypes
###

""" Scalar function field in D-Dimensional space """
abstract type AbstractField{T,D} <: AbstractVector{T} end
""" Operators acting on fields in D-Dimensional space """
abstract type AbstractOperator{T,D} <: AbstractDiffEqOperator{T} end
""" D-Dimensional physical domain """
abstract type AbstractDomain{T,D} end
""" Function space in D-Dimensional space """
abstract type AbstractSpace{T,D} end
""" Boundary condition on domain in D-Dimensional space """
abstract type AbstractBonudaryCondition{T,D} end

AbstractSupertypes{T,D} = Union{
                                AbstractField{T,D},
                                AbstractOperator{T,D},
                                AbstractSpace{T,D},
                                AbstractDomain{T,D},
                                AbstractBonudaryCondition{T,D}
                               }

Base.eltype(::AbstractSupertypes{T,D}) where{T,D} = T
dims(::AbstractSupertypes{T,D}) where{T,D} = D

###
# AbstractOperator subtypes
###

""" Tensor product operator in D-Dimensional space """
abstract type AbstractTensorProductOperator{T,D} <: AbstractOperator{T,D} end

""" Gather-Scatter operator in D-Dimensional space """
abstract type AbstractGatherScatterOperator{T,D} <: AbstractOperator{T,D} end

""" Interpolation operator between D-Dimensional spaces """
abstract type AbstractInterpolationOperator{T,D} <: AbstractOperator{T,D} end

###
# AbstractSpace subtypes
###

""" D-Dimensional spectral space """
abstract type AbstractSpectralSpace{T,D} <: AbstractSpace{T,D} end

""" D-Dimensional tensor-product space """
abstract type AbstractTensorProductSpace{T,D} <: AbstractSpace{T,D} end

# misc
include("utils.jl")
include("NDgrid.jl")

# field
include("Field.jl")

# operators
include("OperatorBasics.jl")
include("Operators.jl")

# domain
include("Domain.jl")

# space
include("Space.jl")
include("DeformSpace.jl")

# polynomial spaces
include("LagrangeMatrices.jl")
#include("LagrangePolynomialSpace.jl")
#include("GalerkinOperators.jl")

# fourier spaces
#include("FourierSpace.jl")

export 
       # fields
       Field,

       # operator conveniences
       IdentityOp, NullOp, AffineOp, ComposeOp, InverseOp, # overload op(u,p,t)

       # Concrete operators
       MatrixOp, DiagonalOp, TensorProductOp2D,

       # Domains
       Interval, BoxDomain,

       # spaces
       LagrangePolynomialSpace,
       GaussLobattoLegendre1D, GaussLegendre1D, GaussChebychev1D,
       GaussLobattoLegendre2D, GaussLegendre2D, GaussChebychev2D

end # module
