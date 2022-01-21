#
""" Tensor Product Polynomial Field """
struct Field{T,N,Tarr <: AbstractArray{T,N}} <: AbstractSpectralField{T,N}
    u::Tarr
end

# printing
function Base.summary(io::IO, u::Field{T,N,Tarr}) where{T,N,Tarr}
    println(io, "$(N)D Tensor Product Polynomial spectral field of type $T")
    Base.show(io, typeof(u))
end
function Base.show(io::IO, ::MIME"text/plain", u::Field{T,N,Tarr}) where{T,N,Tarr}
    iocontext = IOContext(io, :compact => true, :limit => true)
    Base.summary(iocontext, u)
    Base.show(iocontext, MIME"text/plain"(), u.u)
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
    @eval Base.$op(u::Field , v::Number) = $op(u.u, v  ) |> Field
    @eval Base.$op(u::Number, v::Field ) = $op(u  , v.u) |> Field
    if op ∈ (:+, :-,)
        @eval Base.$op(u::Field, v::Field) = $op(u.u, v.u) |> Field
    end
end
Base.:-(u::Field) = Field(-u.u)
Base.:*(u::Adjoint{T,<:Field}, v::Field) where{T} =  dot(u.parent, v)
LinearAlgebra.dot(u::Field, v::Field) = u' * v
LinearAlgebra.norm(u::Field, p::Real=2) = norm(u.u, p)
#
