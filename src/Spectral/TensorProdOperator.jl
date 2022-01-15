#
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

#include("mask.jl")
#include("mass.jl") # boundary operators, etc

#include("lapl.jl")
#incldue("advect")
#incldue("hlmz")

""" ComposeOperator """
""" InverseOperator """
