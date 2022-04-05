#
""" Diagonal Scaling Operator """
struct DiagonalOp{T,N,Tdiag<:Diagonal} <: AbstractOperator{T,N}
  diag::Tdiag
  #
  function DiagonalOp(u::AbstractField{T,N}) where{T,N}
    diag = u |> _vec |> Diagonal
    DiagonalOp(diag, N)
  end
  
 # TODO remove later - DiagonalOp should only support AbstractFields
  function DiagonalOp(u::AbstractArray{T,N}) where{T,N}
    diag = u |> _vec |> Diagonal
    DiagonalOp(diag, N)
  end
  
  function DiagonalOp(diag::Diagonal{T}, N::Integer) where{T}
    new{T,N,typeof(diag)}(diag)
  end
end

Base.size(D::DiagonalOp) = size(D.diag)
Base.adjoint(D::DiagonalOp) = D

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

for op in (
           :+ , :- , :* , :/ , :\ ,
          )
  @eval function Base.$op(A::DiagonalOp{Ta,N,Tadiag},
                          B::DiagonalOp{Tb,N,Tbdiag},
                         ) where{Ta,Tb,N,Tadiag,Tbdiag}
    @assert size(A) == size(B)
    diag = $op(A.diag, B.diag)
    DiagonalOp(diag, N)
  end
end

LinearAlgebra.:lmul!(a::Number,B::DiagonalOp) = lmul!(a,B.diag)
LinearAlgebra.:rmul!(A::DiagonalOp,b::Number) = rmul!(A.diag,b)
Base.inv(D::DiagonalOp{T,N,Tdiag}) where{T,N,Tdiag} = DiagonalOp(inv(D.diag), N)

""" identity operator make it fully lazy like LinearOperators.opEye """
opEye(n::Integer) = opEye(Float64, n)
opEye(T::DataType, n::Integer) = ones(T, n) |> Diagonal

# overload Base.kron, Base.⊗
# do it in 2D, 3D
"""
 Tensor product operator
      (Bs ⊗ Ar) * u
 (Ct ⊗ Bs ⊗ Ar) * u
"""
function tensor_product!(V,U,Ar,Bs,cache) # 2D
  """ V .= Ar * U Bs' """
  mul!(cache, Ar, U)
  mul!(V, cache, Bs')
end

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
  isfresh::Bool
  #
  function TensorProd2DOp(Ar, Bs, cache = nothing,
                          isfresh::Bool = cache === nothing)
    T = promote_type(eltype(Ar), eltype(Bs))
    new{T,typeof(Ar),typeof(Bs),typeof(cache)}(Ar,Bs,cache,isfresh)
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
  if A.isfresh
    cache = init_cache(A, U)
    A = set_cache(A, cache)
  end
end

function LinearAlgebra.mul!(v, A::TensorProd2DOp, u)
  U = u isa AbstractField ? u.array : u
  V = v isa AbstractField ? v.array : v

  if A.isfresh
    cache = init_cache(A, U)
    A = set_cache(A, cache)
    mul!(V, cache, A.Bs') # 2nd half of tensor product computation
    return v
  end

  tensor_product!(V,U,A.Ar,A.Bs,A.cache)
  return v
end
#
