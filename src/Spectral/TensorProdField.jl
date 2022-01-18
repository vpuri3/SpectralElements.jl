#
""" Tensor Product Polynomial Field """
struct Field{T,N,arrT <: AbstractArray{T,N}} <: AbstractSpectralField{T,N}
    u::arrT
end

# printing
function Base.summary(io::IO, u::Field{T,N,arrT}) where{T,N,arrT}
    println(io, "$(N)D Tensor Product Polynomial spectral field of type $T")
    Base.show(io, typeof(u))
end
function Base.show(io::IO, ::MIME"text/plain", u::Field{T,N,arrT}) where{T,N,arrT}
    ioc = IOContext(io, :compact => true, :limit => true)
    Base.summary(ioc, u)
    Base.show(ioc, MIME"text/plain"(), u.u)
    println()
end

# allocation
Base.similar(u::Field) = Field(similar(u.u))
Base.copy(u::Field) = Field(copy(u.u))
function Base.copy!(u::Field, v::Field)
    copy!(u.u,v.u)
    return u
end

# vector indexing
Base.IndexStyle(::Field) = IndexLinear()
Base.getindex(u::Field, i::Int) = getindex(u.u, i)
Base.setindex!(u::Field, v, i::Int) = setindex!(u.u, v, i)
Base.length(u::Field) = length(u.u)
Base.size(u::Field) = (length(u),)

# broadcast
Base.BroadcastStyle(::Type{<:Field}) = Broadcast.ArrayStyle{Field}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{Field}},
                      ::Type{ElType}) where ElType
    u = find_fld(bc)
    Field(similar(Array{ElType}, axes(u.u)))
end

find_fld(bc::Base.Broadcast.Broadcasted) = find_fld(bc.args)
find_fld(args::Tuple) = find_fld(find_fld(args[1]), Base.tail(args))
find_fld(x) = x
find_fld(::Tuple{}) = nothing
find_fld(a::Field, rest) = a
find_fld(::Any, rest) = find_fld(rest)

# math
for op in (
           :+ , :- , :* , :/ , :\ ,
          )
    @eval Base.$op(u::Field, v::Number)   = Field($op(u.u, v)  )
    @eval Base.$op(u::Number  , v::Field) = Field($op(u  , v.u))
    if op ∈ (:+, :-,)
        @eval Base.$op(u::Field, v::Field) = Field($op(u.u, v.u))
    end
end
Base.:-(u::Field) = Field(-u.u)
#Base.:adjoint(u::Field) = u #AdjointField(u)
Base.:*(u::Adjoint{T,<:Field}, v::Field) where{T} =  dot(u.u, v)
LinearAlgebra.dot(u::Field, v::Field) = dot(u.u, v.u)
LinearAlgebra.norm(u::Field, p::Real=2) = norm(u.u, p)
#
