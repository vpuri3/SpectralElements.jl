"""
 the TPPField struct is pretty useless since you can just do
 linsolve(;u=vec(u), b=vec(b))
 vec(u) DOESN'T ALLOCATE!!
 ```julia
 u = ones(3,3);
 v = vec(u);
 v[1] = 0
 u
 3×3 Matrix{Float64}:
  0.0  1.0  1.0
  1.0  1.0  1.0
  1.0  1.0  1.0
 ```

 check out function OrdinaryDiffEq.dolinsolve
 check out https://diffeq.sciml.ai/stable/tutorials/advanced_ode_example/
 still required since we still need to define custom inner product!!

"""
#
""" Tensor Product Polynomial Field """
struct TPPField{T,N,arrT <: AbstractArray{T,N}} <: AbstractSpectralField{T,N}
    u::arrT
end

# printing
function Base.summary(io::IO, u::TPPField{T,N,arrT}) where{T,N,arrT}
    println(io, "$(N)D Tensor Product Polynomial spectral field of type $T")
    Base.show(io, typeof(u))
end
function Base.show(io::IO, ::MIME"text/plain", u::TPPField{T,N,arrT}) where{T,N,arrT}
    ioc = IOContext(io, :compact => true, :limit => true)
    Base.summary(ioc, u)
    Base.show(ioc, MIME"text/plain"(), u.u)
    println()
end

# allocation
Base.similar(u::TPPField) = TPPField(similar(u.u))
Base.copy(u::TPPField) = TPPField(copy(u.u))
function Base.copy!(u::TPPField, v::TPPField)
    copy!(u.u,v.u)
    return u
end

# vector indexing
Base.IndexStyle(::TPPField) = IndexLinear()
Base.getindex(u::TPPField, i::Int) = getindex(u.u, i)
Base.setindex!(u::TPPField, v, i::Int) = setindex!(u.u, v, i)
Base.length(u::TPPField) = length(u.u)
Base.size(u::TPPField) = (length(u),)

# broadcast
Base.BroadcastStyle(::Type{<:TPPField}) = Broadcast.ArrayStyle{TPPField}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TPPField}},
                      ::Type{ElType}) where ElType
    u = find_fld(bc)
    TPPField(similar(Array{ElType}, axes(u.u)))
end

find_fld(bc::Base.Broadcast.Broadcasted) = find_fld(bc.args)
find_fld(args::Tuple) = find_fld(find_fld(args[1]), Base.tail(args))
find_fld(x) = x
find_fld(::Tuple{}) = nothing
find_fld(a::TPPField, rest) = a
find_fld(::Any, rest) = find_fld(rest)

""" Lazy Adjoint Tensor Product Polynomial Field """
#struct AdjointTPPField{T,N,fldT <: TPPField{T,N}} <: AbstractSpectralField{T,N}
#    u::fldT
#end
struct AdjointTPPField{fldT <: TPPField}
    u::fldT
end

# math
for op in (
           :+ , :- , :* , :/ , :\ ,
          )
    @eval Base.$op(u::TPPField, v::Number)   = TPPField($op(u.u, v)  )
    @eval Base.$op(u::Number  , v::TPPField) = TPPField($op(u  , v.u))
    if op ∈ (:+, :-,)
        @eval Base.$op(u::TPPField, v::TPPField) = TPPField($op(u.u, v.u))
    end
end
Base.:-(u::TPPField) = TPPField(-u.u)
Base.:adjoint(u::TPPField) = u #AdjointTPPField(u)
Base.:*(u::AdjointTPPField, v::TPPField) =  dot(u.u, v)
LinearAlgebra.dot(u::TPPField, v::TPPField) = dot(u.u, v.u)
LinearAlgebra.norm(u::TPPField, p::Real=2) = norm(u.u, p)

