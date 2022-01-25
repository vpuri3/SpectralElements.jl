#
""" Common Interface """

Base.*(A::AbstractSpectralOperator, u) = A(u)
function (A::AbstractSpectralOperator)(u) 
  if issquare(A)
    mul!(similar(u),A,u)
  else
    ArgumentError("Operation not defined for $A")
  end
end
function init_cache(A::AbstractSpectralOperator,u)
  cache = nothing
end
function set_cache(A::AbstractSpectralOperator, cache)
  @set! A.cache = cache
  return A
end

"""
Identity Operator

basically LinearAlgebra.I with size()
"""
struct Identity{Ti,Tn}
  Id::Ti
  n ::Tn
end
Base.size(Id::Identity) = (Id.n,Id.n)
Base.eltype(::Identity) = Bool
adjoint(Id::Identity) = Id

#LinearAlgebra.rmul!(A::Identity,b::Number) = rmul!(A.diag,b)
#LinearAlgebra.lmul!(a::Number,B::Identity) = lmul!(a,B.diag)

mul!(v, ::Identity, u) = mul!(v, I, u)
ldiv!(v, ::Identity, u) = ldiv!(v, I, u)
ldiv!(::Identity, u) = ldiv!(I, u)

""" ComposeOperator """
struct ComposeOperator{T,N,Ti,To,Tc} <: AbstractSpectralOperator{T,N}
    inner::Ti
    outer::To
    cache::Tc
    isinit::Bool
    #
    function ComposeOperator(inner::AbstractSpectralOperator{Ti,N},
                             outer::AbstractSpectralOperator{To,N}
                            ) where{Ti,To,N}
        T = promote_type(Ti, To)
        cache  = nothing
        isinit = false
        new{T,N,typeof(inner),typeof(outer),typeof(cache)}(inner, outer, cache, init)
    end
end
(A::ComposeOperator)(u) = A.outer(A.inner(u))
function Base.:∘(outer::AbstractSpectralOperator,
                 inner::AbstractSpectralOperator)
    ComposeOperator(inner,outer)
end

function (A::ComposeOperator)(u)

end
size(A::ComposeOperator) = (size(A.outer, 1), size(A.inner, 2))
adjoint(A::ComposeOperator) = A.inner' ∘ A.outer'

function init_cache(A::TensorProductOp{T,2}, u)
  Ar = A.mats[1]
  cache = Ar * u
  return cache
end

function LinearAlgebra.mul!(y, A::ComposeOperator, x)
  if(A.cache == nothing)
    cache = init_cache(A, x)
    A = set_cache(A, cache)
  end

  mul!(cache, inner, x)
  mul!(y, outer, cache)
end

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
struct InverseOperator{T,N,Ta} <: AbstractSpectralOperator{T,N}
  A::Ta
  #
  function InverseOperator(A::AbstractSpectralOperator{T,N}) where{T,N}
    LinearAlgebra.checksquare(A)
    new{T,N,typeof(A)}(A)
  end
end

inv(A::AbstractSpectralOperator) = InverseOperator(A)
size(A::InverseOperator) = size(A.A)
adjoint(A::InverseOperator) = inv(A.A')
# https://github.com/SciML/LinearSolve.jl/issues/97
LinearAlgebra.ldiv!(A::InverseOperator, x) = mul!(x, A.A, x)
LinearAlgebra.ldiv!(y, A::InverseOperator, x) = mul!(y, A.A, x)
LinearAlgebra.mul!(y, A::InverseOperator, x) = ldiv!(y, A.A, x)

