#
#--------------------------------------#
module Spectral
#--------------------------------------#
import Base: +, -, *, size, getindex, setindex!
import FastGaussQuadrature
using LinearAlgebra

abstract type AbstractSpectralField end
abstract type AbstractSpectralOperator end
abstract type AbstractSpectralSpace end

#--------------------------------------#
export ABu
#--------------------------------------#
"""
 Tensor product operator
 (As ⊗ Br) * u
"""
function ABu(u ::AbstractSpectralField,
             As::AbstractSpectralOperator = I,
             Br::AbstractSpectralOperator = I)

    return B * u * A'
end
#--------------------------------------#
include("GLL.jl")
#--------------------------------------#

end #module
