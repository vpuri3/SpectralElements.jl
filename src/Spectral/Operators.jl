#
## Diagonal Operator

""" Diagonal Scaling Operator """
struct DiagonalOp{T,D,Tdiag<:AbstractField{T,D}} <: AbstractOperator{T,D}
    diag::Tdiag
end

Base.size(A::DiagonalOp) = size(Diagonal(A.diag))
Base.adjoint(A::DiagonalOp) = A

function LinearAlgebra.mul!(v::AbstractField{T,D}, A::DiagonalOp{T,D,Tdiag}, u::AbstractField{T,D}) where{T,D,Tdiag}
    mul!(_vec(v), Diagonal(A.diag), _vec(u))
    return v
end

function LinearAlgebra.ldiv!(v::AbstractField{T,D}, A::DiagonalOp{T,D,Tdiag}, u::AbstractField{T,D}) where{T,D,Tdiag}
    ldiv!(_vec(v), Diagonal(A.diag), _vec(u))
    return v
end

function LinearAlgebra.ldiv!(A::DiagonalOp{T,D,Tdiag}, u::AbstractField{T,D}) where{T,D,Tdiag}
    ldiv!(Diagonal(A.diag), _vec(u))
    return u
end

for op in (
           :+, :-, :*, :/, :\,
          )
  @eval function Base.$op(A::DiagonalOp{Ta,D,Tadiag},
                          B::DiagonalOp{Tb,D,Tbdiag},
                         ) where{Ta,Tb,D,Tadiag,Tbdiag}
    @assert size(A) == size(B)
    diag = $op(A.diag, B.diag)
    DiagonalOp(diag)
  end
end

LinearAlgebra.:lmul!(a::Number,B::DiagonalOp) = lmul!(a,B.diag)
LinearAlgebra.:rmul!(A::DiagonalOp,b::Number) = rmul!(A.diag,b)
Base.inv(A::DiagonalOp{T,D,Tdiag}) where{T,D,Tdiag} = DiagonalOp(inv(A.diag), D)

## Tensor Product Operator
#Base.kron()
#Base.⊗

"""
Tensor product operator

(Bs ⊗ Ar) * u
"""
function tensor_product!(V,U,Ar,Bs,cache) # 2D
    """ V .= Ar * U Bs' """
    mul!(cache, Ar, U)
    mul!(V, cache, Bs')
end

"""
Tensor product operator

(Ct ⊗ Bs ⊗ Ar) * u
"""
function tensor_product!(V,U,Ar,Bs,Ct,cache1,cache2) # 3D
    szU = size(U)
    U_re = _reshape(U, (szU[1], szU[2]*szU[3]))
    mul!(cache1, Ar, U_re)

    # Bs op - write to cache2. use views

    szC = size(cache2)
    C_re = _reshape(cache2, (szC[1]*szC[2], szC[3]))
    mul!(V, C_re, Ct')

    V
end

""" 2D Tensor Product Operator """
struct TensorProd2DOp{T,Ta,Tb,Tc} <: AbstractOperator{T,2}
    Ar::Ta
    Bs::Tb
    #
    cache::Tc
    isunset::Bool
    #
    function TensorProd2DOp(Ar, Bs, cache = nothing,
                            isunset::Bool = cache === nothing)
        T = promote_type(eltype(Ar), eltype(Bs))
        new{T,typeof(Ar),typeof(Bs),typeof(cache)}(Ar,Bs,cache,isunset)
    end
end

Base.size(A::TensorProd2DOp) = size(A.Ar) .* size(A.Bs)
function Base.adjoint(A::TensorProd2DOp)
    if issquare(A)
        TensorProdOp(A.Ar', A.Bs', A.cache)
    else
        TensorProdOp(A.Ar', A.Bs')
    end
end

for op in (
           :+ , :- , :* , :/, :\,
          )
    @eval function Base.$op(A::TensorProd2DOp{Ta,Taa,Tba,Tca},
                            B::TensorProd2DOp{Tb,Tab,Tbb,Tcb}
                           ) where{Ta,Taa,Tba,Tca,Tb,Tab,Tbb,Tcb}
        Ar = $op(A.Ar, B.Ar)
        Bs = $op(A.Bs, B.Bs)
        TensorProd2DOp(Ar, Bs)
    end
end

function init_cache(A::TensorProd2DOp, U)
    cache = A.Ar * U
end

function (A::TensorProd2DOp)(u)
    U = u isa AbstractField ? u.array : u
    if A.isunset
        cache = init_cache(A, U)
        A = set_cache(A, cache)
    end
end

function LinearAlgebra.mul!(v::AbstractField{T,2}, A::TensorProd2DOp{T}, u::AbstractField{T,2}) where{T}

    U = u.array
    V = v.array

    if A.isunset
        cache = init_cache(A, U)
        A = set_cache(A, cache)
        mul!(V, cache, A.Bs')
        return v
    end
    
    tensor_product!(V,U,A.Ar,A.Bs,A.cache)
    return v
end
#
