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

(A::AbstractSpectralOperator)(u) = mul!(similar(u),A,u)

include("TensorProdField.jl")
include("TensorProdOperator.jl")
include("TensorProdSpace.jl")

end #module
