#
""" Common Operator Interface """

# application
function (A::AbstractSpectralOperator)(u) 
  if issquare(A)
    mul!(similar(u),A,u)
  else
    ArgumentError("Operation not defined for $A")
  end
end

Base.:*(A::AbstractSpectralOperator, u::AbstractArray) = A(u)

# fusing
function Base.:*(A::AbstractSpectralOperator, B::AbstractSpectralOperator)
  @error "Fusing operation not defined for $A * $B. Try lazy composition with ∘"
end

# caching
function init_cache(A::AbstractSpectralOperator,u)
  @error "Caching behaviour not defined for $A"
end

function set_cache(A::AbstractSpectralOperator, cache)
  @set! A.cache = cache
  @set! A.isfresh = false
  return A
end

# TODO +, - operations on AbstractOperators
#LinearAlgebra.:rmul!(A::DiagonalOp,b::Number) = rmul!(A.diag,b)
#LinearAlgebra.:lmul!(a::Number,B::DiagonalOp) = lmul!(a,B.diag)

#for op in (
#           :+ , :- , :* , :/ , :\ ,
#          )
#  @eval Base.$op(u::Field , v::Number) = $op(u.array, v) |> Field
#  @eval Base.$op(u::Number, v::Field ) = $op(u, v.array) |> Field
#  if op ∈ (:+, :-,)
#    @eval Base.$op(u::Field, v::Field) = $op(u.array, v.array) |> Field
#  end
#end

"""
Copying Operator

julia> fill(I,(1,2)) * fill(a, (2,))
julia> [op op] * fill(u,(2,))
julia> [op, op] * fill(u,(1,1))
julia> [op, op] * [u]

we just need fill(u, (1,)) and then array math takes care of everything else!

u -> [u, u] where u is AbstractSpectralField
it's adjoint should be summation
"""
struct CopyingOp{T,N,Tdims} <: AbstractSpectralOperator{T,N}
  dims::Tdims
  #
  function CopyingOp(dims, N = length(dims))
    T = Bool
    new{T,N,typeof(dims)}(dims)
  end
end

Base.size(C::CopyingOp) = (Id.n,Id.n)
Base.adjoint(C::CopyingOp) = CopyingOp(reverse(C.dims))
Base.eltype(::CopyingOp) = Bool

(C::CopyingOp)(u) = fill(u,dims)
#mul!(v, ::CopyingOp, u) = mul!(v, I, u)
#ldiv!(v, ::CopyingOp, u) = ldiv!(v, I, u)
#ldiv!(::CopyingOp, u) = ldiv!(I, u)

"""
 figure out caching for composition type problems
 this functionality can work
    applychain(::Tuple{}, x) = x
    applychain(fs::Tuple, x) = applychain(tail(fs), first(fs)(x))
    (c::Chain)(x) = applychain(c.layers, x)

  or use NamedTuple, or NTuple.
"""

""" Lazy Composition """
struct ComposeOperator{T,N,Ti,To,Tc} <: AbstractSpectralOperator{T,N}
  inner::Ti
  outer::To
  #
  cache::Tc
  isfresh::Bool
  #
  function ComposeOperator(inner::AbstractSpectralOperator{Ti,N},
                           outer::AbstractSpectralOperator{To,N},
                           cache = nothing,
                           isfresh::Bool = cache === nothing
                          ) where{Ti,To,N}
    T = promote_type(Ti, To)
    isfresh = cache === nothing
    new{T,N,typeof(inner),typeof(outer),typeof(cache)}(inner, outer, cache, isfresh)
  end
end

function Base.:∘(outer::AbstractSpectralOperator,
                 inner::AbstractSpectralOperator)
  ComposeOperator(inner,outer)
end

Base.size(A::ComposeOperator) = (size(A.outer, 1), size(A.inner, 2))
Base.adjoint(A::ComposeOperator) = A.inner' ∘ A.outer'

function init_cache(A::ComposeOperator, u)
  cache = A.inner(u)
  return cache
end

function (A::ComposeOperator)(u)
  if A.isfresh
    cache = init_cache(A, x)
    A = set_cache(A, cache)
    return outer(cache)
  end
  mul!(cache, inner, u)
  outer(cache)
end

function LinearAlgebra.mul!(y, A::ComposeOperator, x)
  if A.isfresh
    cache = init_cache(A, x)
    A = set_cache(A, cache)
    return mul!(y, outer, cache)
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

Base.inv(A::AbstractSpectralOperator) = InverseOperator(A)
Base.size(A::InverseOperator) = size(A.A)
Base.adjoint(A::InverseOperator) = inv(A.A')
# https://github.com/SciML/LinearSolve.jl/issues/97
LinearAlgebra.ldiv!(A::InverseOperator, x) = mul!(x, A.A, x)
LinearAlgebra.ldiv!(y, A::InverseOperator, x) = mul!(y, A.A, x)
LinearAlgebra.mul!(y, A::InverseOperator, x) = ldiv!(y, A.A, x)
#
