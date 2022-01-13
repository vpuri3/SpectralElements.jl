"""
 the TPPField struct is pretty useless since you can just do
 linsolve(;u=vec(u), b=vec(b))
 vec(u) DOESN'T ALLOCATE!!
 ```julia
 u = ones(3,3);
 v = vec(u);
 v[1] = 0
 u
 3×3 Matrix{Float64}:
  0.0  1.0  1.0
  1.0  1.0  1.0
  1.0  1.0  1.0
 ```

 check out function OrdinaryDiffEq.dolinsolve
 check out https://diffeq.sciml.ai/stable/tutorials/advanced_ode_example/

"""
#
import Base: summary, show                                  # printing
import Base: similar, copy, copy!                           # allocation
import Base: length, size, getindex, setindex!, IndexStyle  # indexing
import Base.Broadcast: BroadcastStyle                       # broadcast
import Base: +, -, *, /, \, adjoint                         # math
import LinearAlgebra: mul!, ldiv!, dot, norm#, axpy!, axpby!, diagonal, Diagonal

abstract type AbstractSpectralField{T,N} <: AbstractVector{T} end
abstract type AbstractSpectralOperator{T,N} end
abstract type AbstractSpectralSpace{T,N} end

#------------------------------------------------------------#
""" Tensor Product Polynomial Field """
struct TPPField{T,N,arrT <: AbstractArray{T,N}} <: AbstractSpectralField{T,N}
    u::arrT
end

# printing
function Base.summary(io::IO, u::TPPField{T,N,arrT}) where{T,N,arrT}
    println(io, "$(N)D Tensor Product Polynomial spectral field of type $T")
    Base.show(io, typeof(u))
end
function Base.show(io::IO, ::MIME"text/plain", u::TPPField{T,N,arrT}) where{T,N,arrT}
    ioc = IOContext(io, :compact => true, :limit => true)
    Base.summary(ioc, u)
    Base.show(ioc, MIME"text/plain"(), u.u)
    println()
end

# allocation
Base.similar(u::TPPField) = TPPField(similar(u.u))
Base.copy(u::TPPField) = TPPField(copy(u.u))
function Base.copy!(u::TPPField, v::TPPField)
    copy!(u.u,v.u)
    return u
end

# vector indexing
Base.IndexStyle(::TPPField) = IndexLinear()
Base.getindex(u::TPPField, i::Int) = getindex(u.u, i)
Base.setindex!(u::TPPField, v, i::Int) = setindex!(u.u, v, i)
Base.length(u::TPPField) = length(u.u)
Base.size(u::TPPField) = (length(u),)

# broadcast
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

""" Lazy Adjoint Tensor Product Polynomial Field """
#struct AdjointTPPField{T,N,fldT <: TPPField{T,N}} <: AbstractSpectralField{T,N}
#    u::fldT
#end
struct AdjointTPPField{fldT <: TPPField}
    u::fldT
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

#------------------------------------------------------------#

""" Diagonal Operator on Tensor Product Polynomial field """
struct TPPDiagOp{T,N,fldT<:TPPField{T,N}} <: AbstractSpectralOperator{T,N}
    u::fldT
    #
    TPPDiagOp(u::TPPField{T,N,fldT}) where{T,N,fldT} = new{T,N,typeof(u)}(u)
    TPPDiagOp(u::AbstractArray{T,N}) where{T,N} = TPPDiagOp(TPPField(u))
end
eltype(D::TPPDiagOp) = eltype(D.u)
function size(D::TPPDiagOp{T,N,fldT}) where{T,N,fldT}
    l = length(D.u)
    return (l,l)
end
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

# printing
function Base.summary(io::IO, D::TPPDiagOp{T,N,fldT}) where{T,N,fldT}
    println(io, "Diagonal operator on $(N)D Tensor Product Polynomial spectral field of type $T")
    Base.show(io, typeof(D))
end
function Base.show(io::IO, ::MIME"text/plain", D::TPPDiagOp{T,N,fldT}) where{T,N,fldT}
    ioc = IOContext(io, :compact => true, :limit => true)
    Base.summary(ioc, D)
    Base.show(ioc, MIME"text/plain"(), D.u.u)
    println()
end

#------------------------------------------------------------#

"""
 Tensor product operator
      (Bs ⊗ Ar) * u
 (Ct ⊗ Bs ⊗ Ar) * u
"""
function TensorProduct(u ::TPPField{T,2},
                       Ar::AbstractMatrix = I,
                       Bs::AbstractMatrix = I,
    ) where{T}
    u = u.u
    return TPPField(Ar * u * Bs')
end

function TensorProduct(u ::TPPField{T,3},
                       Ar::AbstractMatrix = I,
                       Bs::AbstractMatrix = I,
                       Ct::AbstractMatrix = I,
    ) where{T}
    v = u
    return TPPField(v)
end

""" Tensor Product Operator on Tensor Product Polynomial Filed """
struct TPPOp{T,N,matT <: AbstractMatrix{T}} <: AbstractSpectralOperator{T,N}
    A::matT
    #
    TPPOp(A::AbstractMatrix{T},N::Int=2) where{T} = new{T,N,typeof(A)}(A)
end
eltype(D::TPPOp) = eltype(D.A)
size(D::TPPOp{T,N,matT}) where{T,N,matT} = Tuple(size(D.A,1) for i=1:N)
#adjoint(D::TPPOp) = D
#(D::TPPOp)(u::TPPField) = TPPField(D.u .* u)

include("derivMat.jl")
include("interp.jl")

export GLL2D
""" Gauss Lobatto Legendre spectral space """
struct GLL2D{T,vecT,fldT,matT,F} <: AbstractSpectralSpace{T,2}
    z::vecT
    w::vecT

    r::TPPField{T,2,fldT}  # reference coordinates
    s::TPPField{T,2,fldT}

    x::TPPField{T,2,fldT}  # spatial coordinates
    y::TPPField{T,2,fldT}

    B::TPPDiagOp{2,T,fldT} # mass   matrix
    D::TPPOp{2,T,matT}     # deriv  matrix
#   J::TPPOp{2,T,matT}     # interp matrix

    deform::F
end

function GLL2D(n::Int = 8,
               deform = (r,s) -> (copy(r), copy(s)),
               T=Float64)
    z,w = FastGaussQuadrature.gausslobatto(n)
    z = T.(z)
    w = T.(w)

    o = ones(T,n)
    r = z * o' |> TPPField
    s = o * z' |> TPPField

    x,y = deform(r,s)

    B = w * w'      |> TPPDiagOp
    D = derivMat(z) |> TPPOp

    return GLL2D(z,w,r,s,x,y,B,D, deform)
end

#include("mask.jl")
#include("mass.jl") # boundary operators, etc
#
#include("lapl.jl")
#incldue("advect")
#incldue("hlmz")
