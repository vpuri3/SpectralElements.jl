#
import Base: similar, copy, copy!                           # allocation
import Base: length, size, getindex, setindex!, IndexStyle  # indexing
import Base: +, -, *, /, \, adjoint                         # math
import LinearAlgebra: mul!, ldiv!, dot, norm#, axpy!, axpby!, diagonal, Diagonal
import Base.Broadcast: BroadcastStyle

abstract type AbstractSpectralField{T,N} <: AbstractVector{T} end
abstract type AbstractSpectralOperator{T,N} end
abstract type AbstractSpectralSpace{T,N} end

#------------------------------------------------------------#
""" Tensor Product Polynomial Field """
struct TPPField{T,N,arrT <: AbstractArray{T,N}} <: AbstractSpectralField{T,N}
    u::arrT
end

""" Lazy Adjoint Tensor Product Polynomial Field """
#struct AdjointTPPField{T,N,fldT <: TPPField{T,N}} <: AbstractSpectralField{T,N}
#    u::fldT
#end
struct AdjointTPPField{fldT <: TPPField}
    u::fldT
end

# vector indexing
Base.IndexStyle(::TPPField) = IndexLinear()
Base.getindex(u::TPPField, i::Int) = getindex(u.u, i)
Base.setindex!(u::TPPField, v, i::Int) = setindex!(u.u, v, i)
Base.length(u::TPPField) = length(u.u)
Base.size(u::TPPField) = (length(u), 1)

# allocation
Base.similar(u::TPPField) = TPPField(similar(u.u))
Base.copy(u::TPPField) = TPPField(copy(u.u))
function Base.copy!(u::TPPField, v::TPPField)
    copy!(u.u,v.u)
    return u
end

# math
for op in (
           :+ , :- , :* , :/ , :\ ,
          )
    @eval Base.$op(u::TPPField, v::Number)   = TPPField($op(u.u, v)  )
    @eval Base.$op(u::Number  , v::TPPField) = TPPField($op(u  , v.u))
    if op ∈ (:+, :-,)
        @eval Base.$op(u::TPPField, v::TPPField) = TPPField($op(u.u, v.u))
    end
end
Base.:-(u::TPPField) = TPPField(-u.u)
Base.:adjoint(u::TPPField) = u #AdjointTPPField(u)
Base.:*(u::AdjointTPPField, v::TPPField) =  dot(u.u, v)
LinearAlgebra.dot(u::TPPField, v::TPPField) = dot(u.u, v.u)
LinearAlgebra.norm(u::TPPField, p::Real=2) = norm(u.u, p)

# implement custom broadcasting
Base.BroadcastStyle(::Type{<:TPPField}) = Broadcast.ArrayStyle{TPPField}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TPPField}},
                      ::Type{ElType}) where ElType
    u = find_fld(bc)
    TPPField(similar(Array{ElType}, axes(u.u)))
end

find_fld(bc::Base.Broadcast.Broadcasted) = find_fld(bc.args)
find_fld(args::Tuple) = find_fld(find_fld(args[1]), Base.tail(args))
find_fld(x) = x
find_fld(::Tuple{}) = nothing
find_fld(a::TPPField, rest) = a
find_fld(::Any, rest) = find_fld(rest)
#------------------------------------------------------------#

""" Diagonal Operator on Tensor Product Polynomial field """
struct TPPDiagOp{T,N,fldT<:TPPField{T,N}} <: AbstractSpectralOperator{T,N}
    u::fldT
    #
    TPPDiagOp(u::TPPField{T,N,fldT}) where{T,N,fldT} = new{T,N,typeof(u)}(u)
    TPPDiagOp(u::AbstractArray{T,N}) where{T,N} = TPPDiagOp(TPPField(u))
end
eltype(D::TPPDiagOp) = eltype(D.u)
size(D::TPPDiagOp{T,N,fldT}) where{T,N,fldT} = Tuple(length(D.u) for i=1:N)
adjoint(D::TPPDiagOp) = D
(D::TPPDiagOp)(u::TPPField) = TPPField(D.u .* u)

function LinearAlgebra.mul!(v::TPPField, D::TPPDiagOp, u::TPPField)
    @. v.u = D.u * u.u
    return v
end

function LinearAlgebra.ldiv!(v::TPPField, D::TPPDiagOp, u::TPPField)
    @. v.u = D.u \ u.u
    return v
end

function LinearAlgebra.ldiv!(D::TPPDiagOp, u::TPPField)
    @. u.u /= D.u
    return u
end

#------------------------------------------------------------#

"""
 Tensor product operator
      (Bs ⊗ Ar) * u
 (Ct ⊗ Bs ⊗ Ar) * u
"""
function TensorProduct(u::TPPField{T,2}, Ar = I, Bs = I) where{T,N}
    return Ar * u * Bs'
end

function TensorProduct(u::TPPField{T,3}, Ar = I, Bs = I, Ct = I) where{T,N}
    return Ar * u * Bs'
end

""" Tensor Product Operator on Tensor Product Polynomial Filed """
struct TPPOp{T,N,matT <: AbstractMatrix{T}} <: AbstractSpectralOperator{T,N}
    A::matT
end
eltype(D::TPPOp) = eltype(D.A)
#size(D::TPPOp{T,N,matT}) where{T,N,arrT} = Tuple(size(D.A,1) for i=1:N)
#adjoint(D::TPPOp) = D
#(D::TPPOp)(u::TPPField) = TPPField(D.u .* u)

include("derivMat.jl")
include("interp.jl")

#=
""" Gauss Lobatto Legendre spectral space """
export TPP2D
struct TPP2D{T} <: AbstractSpectralSpace{T,2}
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

=#
