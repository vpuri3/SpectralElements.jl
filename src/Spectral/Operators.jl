###
# Diagonal Operator
###

""" Diagonal Scaling Operator """
struct DiagonalOp{T,D,Tdiag<:AbstractField{T,D}} <: AbstractOperator{T,D}
    diag::Tdiag
end

SciMLBase.has_ldiv(::DiagonalOp) = true
SciMLBase.has_ldiv!(::DiagonalOp) = true

Base.size(A::DiagonalOp) = size(Diagonal(A.diag))
Base.adjoint(A::DiagonalOp) = A

function Base.\(A::DiagonalOp{Ta,D}, u::AbstractField{Tu,D}) where{Ta,Tu,D}
    Diagonal(A.diag) \ _vec(u)
end

function LinearAlgebra.mul!(v::AbstractField{Tv,D}, A::DiagonalOp{Ta,D}, u::AbstractField{Tu,D}) where{Tu,Ta,Tv,D}
    mul!(_vec(v), Diagonal(A.diag), _vec(u))
    return v
end

function LinearAlgebra.ldiv!(v::AbstractField{Tv,D}, A::DiagonalOp{Ta,D}, u::AbstractField{Tu,D}) where{Tv,Ta,Tu,D}
    ldiv!(_vec(v), Diagonal(A.diag), _vec(u))
    return v
end

function LinearAlgebra.ldiv!(A::DiagonalOp{Ta,D}, u::AbstractField{Tu,D}) where{Ta,Tu,D}
    ldiv!(Diagonal(A.diag), _vec(u))
    return u
end

# fusion
for op in (
           :+, :-, :*, :/, :\,
          )
    @eval function Base.$op(A::DiagonalOp{Ta,D,Tadiag},
                            B::DiagonalOp{Tb,D,Tbdiag},
                           ) where{Ta,Tb,D,Tadiag,Tbdiag}
        Diag = $op(Diagonal(A.diag), Diagonal(B.diag))
        DiagonalOp(Diag.diag)
    end

    @eval function Base.$op(λ::Number, A::DiagonalOp)
        diag = $op(λ, A.diag)
        DiagonalOp(diag)
    end

    @eval function Base.$op(A::DiagonalOp, λ::Number)
        diag = $op(A.diag, λ)
        DiagonalOp(diag)
    end
end

LinearAlgebra.:lmul!(a::Number,B::DiagonalOp) = lmul!(a,B.diag)
LinearAlgebra.:rmul!(A::DiagonalOp,b::Number) = rmul!(A.diag,b)

###
# Tensor Product Operator
###

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
struct TensorProduct2DOp{T,
                         Ta <: AbstractOperator{<:Number,1},
                         Tb <: AbstractOperator{<:Number,1},
                         Tc
                        } <: AbstractTensorProductOperator{T,2}
    A::Ta
    B::Tb

    cache::Tc
    isunset::Bool

    function TensorProduct2DOp(A, B, cache = nothing, isunset = cache === nothing)
        T = promote_type(eltype(A), eltype(B))
        new{T,typeof(A),typeof(B),typeof(cache)}(A, B, cache, isunset)
    end
end


Base.size(A::TensorProduct2DOp) = size(A.Ar) .* size(A.Bs)

function Base.*(A::TensorProduct2DOp{Ta,D}, u::AbstractField{Tu,D}) where{Ta,Tu,D}
    v = copy(u)
    @set! v.array = A.B * u.array * A.A'
end

function Base.adjoint(A::TensorProduct2DOp)
    if issquare(A)
        TensorProdOp(A.A', A.B', A.cache)
    else
        TensorProdOp(A.A', A.B')
    end
end

for op in (
           :+ , :- , :* , :/, :\,
          )
    @eval function Base.$op(A::TensorProduct2DOp{Ta,Taa,Tba,Tca},
                            B::TensorProduct2DOp{Tb,Tab,Tbb,Tcb}
                           ) where{Ta,Taa,Tba,Tca,Tb,Tab,Tbb,Tcb}
        Ar = $op(A.Ar, B.Ar)
        Bs = $op(A.Bs, B.Bs)
        TensorProduct2DOp(Ar, Bs)
    end
end

function init_cache(A::TensorProduct2DOp, U)
    cache = A.Ar * U
end

function (A::TensorProduct2DOp)(u)
    U = u isa AbstractField ? u.array : u
    if A.isunset
        cache = init_cache(A, U)
        A = set_cache(A, cache)
    end
end

function LinearAlgebra.mul!(v::AbstractField{T,2}, A::TensorProduct2DOp{T}, u::AbstractField{T,2}) where{T}
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
