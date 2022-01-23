#
""" Diagonal Operator on Tensor Product Polynomial field """
struct DiagonalOp{T,N,Tdiag} <: AbstractSpectralOperator{T,N}
    diag::Tdiag
end

function DiagonalOp(u::AbstractSpectralField{T,N}) where{T,N}
    D = u |> _vec |> Diagonal
    DiagonalOp{T,N,typeof(D)}(D)
end

function DiagonalOp(u::AbstractArray{T,N}) where{T,N}
    D = u |> _vec |> Diagonal
    DiagonalOp{T,N,typeof(D)}(D)
end

size(D::DiagonalOp) = size(D.diag)
adjoint(D::DiagonalOp) = D

function *(A::DiagonalOp{Ta,N,Tadiag},
           B::DiagonalOp{Tb,N,Tbdiag},
          ) where{Ta,N,Tadiag,Tb,Tbdiag}
    diag = A.diag .* B.diag
    T = promote_type(Ta,Tb)
    DiagonalOp{T,N,typeof(diag)}(diag)
end

function LinearAlgebra.mul!(v, D::DiagonalOp, u)
    mul!(_vec(v),D.diag,_vec(u))
    return v
end

function LinearAlgebra.ldiv!(v, D::DiagonalOp, u)
    ldiv!(_vec(v),D.diag,_vec(u))
    return v
end

function LinearAlgebra.ldiv!(D::DiagonalOp, u)
    ldiv!(D.diag,_vec(u))
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

"""
 figure out caching

 this functionality can work
    applychain(::Tuple{}, x) = x
    applychain(fs::Tuple, x) = applychain(tail(fs), first(fs)(x))
    (c::Chain)(x) = applychain(c.layers, x)

  or use NamedTuple, or NTuple.
  Kronecker.jl
"""

"""
 Tensor product operator
      (Bs ⊗ Ar) * u
 (Ct ⊗ Bs ⊗ Ar) * u
"""
function tensor_product!(V,U,Ar,Bs,C) # 2D
    """ V .= Ar * U Bs' """
    mul!(C, Ar, U)
    mul!(V, C, Bs')
end

function tensor_product!(V,U,Ar,Bs,Ct,C1,C2)
#   for k=1:size(U,3)
#       @views tensor_product2D(C1[:,:,k],U[:,:,k],Ar,Bs,C1)
#   end
#   mul!(C2,C1,)
#   mul!(V)
    V
end

struct TensorProductOp{T,N,Tm<:Tuple,Tc<:Tuple} <: AbstractSpectralOperator{T,N}
    mats::Tm
    cache::Tc
    #
    function TensorProductOp(mats::Tuple,cache::Tuple)
        T = promote_type(eltype.(mats)...)
        N = length(mats)

        new{T,N,typeof(mats),typeof(cache)}(mats, cache)
    end
    function TensorProductOp(As, Br,u)
        U = u isa AbstractSpectralField ? u.array : u
        mats = As, Br
        cache = begin
            ma, na = size(Ar)
            mb, nb = size(Bs)
            (Ar*U,)
        end
        TensorProductOp(mats,cache)
    end
    function TensorProductOp(As, Br, Ct, u)
        U = u isa AbstractSpectralField ? u.array : u
        mats = As, Br, Ct
        cache = begin
            ma, na = size(Ar)
            mb, nb = size(Bs)
            mc, nc = size(Ct)
            (nothing,)
        end
        TensorProductOp(mats,cache)
    end
end
adjoint(A::TensorProductOp) = TensorProductOp(adjoint.(A.mats),adjoint.(A.cache))
size(A::TensorProductOp) = @. *(size(A.mats)...)

function LinearAlgebra.mul!(v, A::TensorProductOp{T,N,Tm,Tc}, u) where{T,N,Tm,Tc}
    Ar, Bs = A.mats[1:2]
    C = A.cache[1]

    U = u isa AbstractSpectralField ? u.array : u
    V = v isa AbstractSpectralField ? v.array : v

    tensor_product!(V,U,Ar,Bs,C)
    return v
end

function *(A::TensorProductOp{Ta,N,Tam,Tac},
           B::TensorProductOp{Tb,N,Tbm,Tbc}
          ) where{Ta,N,Tam,Tac,Tb,Tbm,Tbc}
    mats = ( A.mats[i] * B.mats[i] for i=1:N)
    TensorProductOp(mats)
end
#
