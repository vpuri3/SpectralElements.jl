#
""" Diagonal Operator on Tensor Product Polynomial field """
#DiagonalOp(u::AbstractArray) = u |> vec |> Diagonal
# just works!! now make it subtype of AbstractSpectralOperator

struct DiagonalOp{T,N,diagT} <: AbstractSpectralOperator{T,N}
    D::diagT
    #
    function DiagonalOp(u::AbstractSpectralField{T,N}) where{T,N}
        D = u |> vec |> Diagonal
        new{T,N,typeof(D)}(u)
    end
    function DiagonalOp(u::AbstractArray{T,N}) where{T,N}
        D = u |> vec |> Diagonal
        new{T,N,typeof(D)}(u)
    end
end
size(D::DiagonalOp) = size(D.D)
adjoint(D::DiagonalOp) = D

function LinearAlgebra.mul!(v, D::DiagonalOp, u)
    mul!(vec(v),D.D,vec(u))
    return v
end

function LinearAlgebra.ldiv!(v, D::DiagonalOp, u)
    ldiv!(vec(v),D.D,vec(u))
    return v
end

function LinearAlgebra.ldiv!(D::DiagonalOp, u)
    ldiv!(D.D,vec(u))
    return u
end

# printing
function Base.summary(io::IO, D::DiagonalOp{T,N,diagT}) where{T,N,diagT}
    println(io, "Diagonal operator on $(N)D Tensor Product Polynomial ",
                "spectral field of type $T")
    Base.show(io, typeof(D))
end
function Base.show(io::IO, ::MIME"text/plain", D::DiagonalOp{T,N,diagT}) where{T,N,diagT}
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
function TensorProduct(u ::Field{T,2},
                       Ar::AbstractMatrix = I,
                       Bs::AbstractMatrix = I,
    ) where{T}
    u = u.u
    return Field(Ar * u * Bs')
end

function TensorProduct(u ::Field{T,3},
                       Ar::AbstractMatrix = I,
                       Bs::AbstractMatrix = I,
                       Ct::AbstractMatrix = I,
    ) where{T}
    v = u
    return Field(v)
end

"""
MiscOp
- MiscOp{T,N} <: AbstractSpectralOperator{T,N}
- implement
  constructor (for AbstractSpectralField, AbstractArray) , size, length,
  adjoint, mul!, ldiv! <-- efficient, nonallocating!
  (op::MiscOp)(u::Field) can be expensive/allocating

TensorProdOp
- figure out size, length, cheapest way to construct TPPOp(Ar,I), TPPOp(I,Bs)
"""

""" Tensor Product Operator on Tensor Product Polynomial Filed """
struct TensorProdOp{T,N,matT} <: AbstractSpectralOperator{T,N}
    Ar::matT
#   Bs::matT
#   Ct::matT
    #
    TensorProdOp(A::AbstractMatrix{T},N::Int=2) where{T} = new{T,N,typeof(A)}(A)
end
#size(D::TensorProdOp{T,N,matT}) where{T,N,matT} = Tuple(size(D.A,1) for i=1:N)
#adjoint(A::TensorProdOp) = TensorProdOp(A.Ar,A.Bs,A.Ct)
#(D::TensorProdOp)(u::Field) = Field(D.u .* u)

include("derivMat.jl")
include("interp.jl")

#include("mask.jl")
#include("mass.jl") # boundary operators, etc

#include("lapl.jl")
#incldue("advect")
#incldue("hlmz")

# LienarSolve/preconditioners.jl
""" ComposeOperator """
struct ComposeOperator{T,N,Ti,To} <: AbstractSpectralOperator{T,N}
    inner::Ti
    outer::To
    #
    function ComposeOperator(inner::AbstractSpectralOperator{T,N},
                             outer::AbstractSpectralOperator{T,N}) where{T,N}
        return new{T,N,typeof(inner),typeof(outer)}(inner,outer)
    end
end
(c::ComposeOperator)(u) = c.outer(c.inner(u))
function Base.:∘(outer::AbstractSpectralOperator,
                 inner::AbstractSpectralOperator)
    return ComposeOperator(inner,outer)
end
size(c::ComposeOperator) = (size(c.outer, 1), size(c.inner, 2))
eltype(c::ComposeOperator) = promote_type(eltype(c.inner), eltype(c.outer))

""" InverseOperator """
struct InverseOperator{T,N,opT} <: AbstractSpectralOperator{T,N}
    A::opT
    #
    function InverseOperator(A::AbstractSpectralOperator{T,N}) where{T,N}
        return new{T,N,typeof(A)}(A)
    end
end

""" Gradient Operator """
struct Gradient{T,N}
end

""" Laplace Operator """
struct Laplace{T,N,fldT}
    G::Matrix{fldT}
end

""" Convection Operator """
struct Convection{T,N,fldT}
    v::fldT
end
#
