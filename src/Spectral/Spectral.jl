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

include("misc_utils.jl")
include("OperatorBasics.jl")
include("TensorProdField.jl")
include("TensorProdOperator.jl")
include("DerivMat.jl")
include("InterpMat.jl")
#include("TensorProdSpace.jl")

export Identity,
       Field,
       CopyingOp, ComposeOperator, InverseOperator,
       DiagonalOp, TensorProd2DOp,
       SpectralSpace2D, GaussLobattoLegendre2D, GaussLegendre2D, GaussChebychev2D

end # module
