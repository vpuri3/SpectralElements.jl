#
""" utilize Base.ReshaedArray """
_reshape(a,dims::NTuple{N,Int}) where{N} = reshape(a,dims) # fallback
_reshape(a::Array, dims::NTuple{N,Int}) where{N} = Base.ReshapedArray(a, dims, ())
#
_vec(a::AbstractVector) = a
_vec(a::AbstractArray) = _reshape(a,(length(a),))

""" check if operator(s) is square """
issquare(::UniformScaling) = true
issquare(A::AbstractMatrix) = size(A,1) === size(A,2)
issquare(A...) = @. (&)(issquare(A)...)

""" Identity Matrix with the notion of size """
struct Identity{Ti,Tn} #<: AbstractMatrix{Bool}
  I::Ti
  n::Tn
  #
  function Identity(n)
    id = I
    new{typeof(id),typeof(n)}(id,n)
  end
end
Base.size(Id::Identity) = (Id.n, Id.n)
Base.eltype(::Identity) = Bool
Base.adjoint(Id::Identity) = Id
#
(::Identity)(u) = u
(*)(::Identity, u) = copy(u) # unnecessary if AbstractMatrix
LinearAlgebra.mul!(v, id::Identity, u) = mul!(v, id.I, u)
LinearAlgebra.ldiv!(v, id::Identity, u) = ldiv!(v, id.I, u)
LinearAlgebra.ldiv!(id::Identity, u) = ldiv!(id.I, u)

