#
""" utilize Base.ReshaedArray """
_reshape(a,dims::NTuple{N,Int}) where{N} = reshape(a,dims)
_reshape(a::Array, dims::NTuple{N,Int}) where{N} = Base.ReshapedArray(a, dims, ())

_vec(a) = vec(a)
_vec(a::AbstractVector) = a
_vec(a::AbstractArray) = _reshape(a,(length(a),))

""" check if operator(s) is square """
issquare(A) = size(A,1) === size(A,2)
issquare(A...) = @. (&)(issquare(A)...)
issquare(::UniformScaling) = true
#issquare(::Identity) = true
#issquare(::DiagonalOp) = true
#
