
include("derivMat.jl")

"""
 Gauss-Lobatto-Legendre spectral field
"""
mutable struct GLLField{T,N} <: AbstractSpectralField{T,N}
    u::AbstractArray{T,N}
end

for op in (:+, :-, :*, :/, :\)
    @eval Base.$op(u::GLLField, args...) = GLLField($op(u.u, args...))
end

#Base.size(u::GLLField, args...) = size(u.u, args...)
for op in (:size, :getindex, :setindex!,)
    @eval Base.$op(u::GLLField, args...) = Base.$op(u.u, args...)
end

"""
 Diagonal mass matrix
"""
mutable struct Mass{T} <: AbstractSpectralOperator{T}
    B::AbstractArray{T}
end
size(B::Mass) = (length(B.B),length(B.B))
adjoint(B::Mass) = copy(B)
(B::Mass)(u::GLLField) = B.B .* u

"""
 Derivative operator
"""
mutable struct Deriv{T} <: AbstractSpectralOperator{T}
    D::AbstractMatrix{T}
end
size(D::Deriv) = (length(D.D),length(D.D))
adjoint(D::Deriv) = Deriv(D.D')
(D::Deriv)(u::GLLField) = D.D * u

export GLL2D
"""
 Gauss Lobatto Legendre spectral space
"""
mutable struct GLL2D{T} <: AbstractSpectralSpace{T,2}
    z::AbstractVector{T}
    w::AbstractVector{T}

    r::GLLField{T,2}
    s::GLLField{T,2}

    B::Mass{T}
    D::Deriv{T}
end

function GLL2D(n,T=Float64)
    z,w = FastGaussQuadrature.gausslobatto(n)

    o = ones(n)
    r = z * o' |> GLLField
    s = o * z' |> GLLField

    B = w * w'      |> Mass
    D = derivMat(z) |> Deriv

    return GLL2D{T}(z,w,r,s,B,D)
end

#include("mask.jl")
#include("mass.jl") # boundary operators, etc
#
#include("lapl.jl")
#incldue("advect")
#incldue("hlmz")

