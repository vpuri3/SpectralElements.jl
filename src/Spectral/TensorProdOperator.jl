#
""" Diagonal Operator on Tensor Product Polynomial field """
#DiagonalOp(u::AbstractArray) = u |> vec |> Diagonal
# just works!! now make it subtype of AbstractSpectralOperator

struct DiagonalOp{T,N,Tdiag} <: AbstractSpectralOperator{T,N}
    diag::Tdiag
    #
#   function DiagonalOp(u::AbstractSpectralField{T,N}) where{T,N}
#       D = u |> vec |> Diagonal
#       new{T,N,typeof(D)}(D)
#   end
#   function DiagonalOp(u::AbstractArray{T,N}) where{T,N}
#       D = u |> vec |> Diagonal
#       new{T,N,typeof(D)}(D)
#   end
    function DiagonalOp(u::Union{AbstractArray{T,N},
                                 AbstractSpectralField{T,N} }
                       ) where{T,N}
        diag = u |> vec |> Diagonal
        new{T,N,typeof(diag)}(diag)
    end
end
size(D::DiagonalOp) = size(D.diag)
adjoint(D::DiagonalOp) = D

function *(A::DiagonalOp{Ta,N,Tadiag},
           B::DiagonalOp{Tb,N,Tbdiag}
          ) where{Ta,Tadiag,Tb,Tbdiag}
    diag = A.diag * B.diag
    TensorProductOp(diag)
end

function LinearAlgebra.mul!(v, D::DiagonalOp, u)
    mul!(vec(v),D.diag,vec(u))
    return v
end

function LinearAlgebra.ldiv!(v, D::DiagonalOp, u)
    ldiv!(vec(v),D.diag,vec(u))
    return v
end

function LinearAlgebra.ldiv!(D::DiagonalOp, u)
    ldiv!(D.diag,vec(u))
    return u
end

# printing
function Base.summary(io::IO, D::DiagonalOp{T,N,Tdiag}) where{T,N,Tdiag}
    println(io, "Diagonal operator on $(N)D Tensor Product Polynomial ",
                "spectral field of type $T")
    Base.show(io, typeof(D))
end

function Base.show(io::IO, ::MIME"text/plain", D::DiagonalOp{T,N,Tdiag}) where{T,N,Tdiag}
    iocontext = IOContext(io, :compact => true, :limit => true)
    Base.summary(iocontext, D)
    Base.show(iocontext, MIME"text/plain"(), D.diag)
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

struct TensorProductOp{T,N,Tmats <:Tuple, Tc} <: AbstractSpectralOperator{T,N}
    mats::Tmats # = (Ar, Bs, Ct)
    cache::Tc
    #
    function TensorProductOp(mats...)
        T = promote_type(eltype.(mats)...)
        N = length(mats)

        cache = nothing
        new{T,N,typeof(mats),typeof(cache)}(mats, cache)
    end
end

(A::TensorProductOp)(u) = TensorProduct(u,A.mats...)
adjoint(A::TensorProductOp) = TensorProductOp(adjoint.(A.mats)..., A.cache)
size(A::TensorProductOp) = @. *(size(A.mats)...)

function *(A::TensorProductOp{Ta,N,Tam,Tac},
           B::TensorProductOp{Tb,N,Tbm,Tbc}
          ) where{Ta,Tam,Tac,Tb,Tbm,Tbc}
    mats = ( A.mats[i] * B.mats[i] for i=1:N)
    TensorProductOp(mats)
end

"""
 figure out caching
 have exception for LinearAlgebra.checksquare(mats...)
 if all square operations then caching isn't necessary. can do just mul!
 if interpolating (like bw vel and pres grids), then cache intermediate arrays

 this functionality can work
    applychain(::Tuple{}, x) = x
    applychain(fs::Tuple, x) = applychain(tail(fs), first(fs)(x))
    (c::Chain)(x) = applychain(c.layers, x)

  or use NamedTuple, or NTuple.
  Kronecker.jl
"""

function LinearAlgebra.mul!(v, A::TensorProductOp{T,N,Tm}, u) where{T,N,Tm}
    return v
end

function LinearAlgebra.ldiv!(v, A::TensorProductOp{T,N,Tm}, u) where{T,N,Tm}
    return v
end

function LinearAlgebra.ldiv!(A::TensorProductOp{T,N,Tm}, u) where{T,N,Tm}
    return u
end

""" ComposeOperator """
struct ComposeOperator{T,N,Ti,To} <: AbstractSpectralOperator{T,N}
    inner::Ti
    outer::To
    #
    function ComposeOperator(inner::AbstractSpectralOperator{Ti,N},
                             outer::AbstractSpectralOperator{To,N}
                            ) where{Ti,To,N}
        T = promote_type(Ti, To)
        new{T,N,typeof(inner),typeof(outer)}(inner,outer)
    end
end
(A::ComposeOperator)(u) = A.outer(A.inner(u))
function Base.:∘(outer::AbstractSpectralOperator,
                 inner::AbstractSpectralOperator)
    ComposeOperator(inner,outer)
end
size(A::ComposeOperator) = (size(A.outer, 1), size(A.inner, 2))

function LinearAlgebra.ldiv!(A::ComposeOperator, x)
    @unpack inner, outer = A

    ldiv!(inner, x)
    ldiv!(outer, x)
end

function LinearAlgebra.ldiv!(y, A::ComposeOperator, x)
    @unpack inner, outer = A

    ldiv!(y, inner, x)
    ldiv!(outer, y)
end

""" InverseOperator """
struct InverseOperator{T,N,opT} <: AbstractSpectralOperator{T,N}
    A::opT
    #
    function InverseOperator(A::AbstractSpectralOperator{T,N}) where{T,N}
        new{T,N,typeof(A)}(A)
    end
end

inv(A::AbstractSpectralOperator) = InverseOperator(A)
# https://github.com/SciML/LinearSolve.jl/issues/97
LinearAlgebra.ldiv!(A::InverseOperator, x) = mul!(x, A.A, x)
LinearAlgebra.ldiv!(y, A::InverseOperator, x) = mul!(y, A.A, x)
LinearAlgebra.mul!(y, A::InverseOperator, x) = ldiv!(y, A.A, x)

#include("derivMat.jl")
#include("interp.jl")

#include("mask.jl")
#include("mass.jl") # boundary operators, etc

#include("lapl.jl")
#incldue("advect")
#incldue("hlmz")

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
