#
#TODO make sure these are fine: @inline, @propagate_inbounds, @boundscheck
#
# ref - https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array

""" Tensor Product Polynomial Field """
struct Field{T,N,Tarr <: AbstractArray{T,N}} <: AbstractSpectralField{T,N}
    array::Tarr
end

# printing
function Base.summary(io::IO, u::Field{T,N,Tarr}) where{T,N,Tarr}
    println(io, "$(N)D Tensor Product Polynomial spectral field of type $T")
    Base.show(io, typeof(u))
end
function Base.show(io::IO, ::MIME"text/plain", u::Field{T,N,Tarr}) where{T,N,Tarr}
    iocontext = IOContext(io, :compact => true, :limit => true)
    Base.summary(iocontext, u)
    Base.show(iocontext, MIME"text/plain"(), u.array)
    println()
end

# vector indexing
Base.IndexStyle(::Field) = IndexLinear()
Base.getindex(u::Field, i::Int) = getindex(u.array, i)
Base.setindex!(u::Field, v, i::Int) = setindex!(u.array, v, i)
Base.size(u::Field) = (length(u.array),)

# allocation
Base.similar(u::Field, ::Type{T} = eltype(u), dims::Dims = size(u.array)) where{T} = Field(similar(u.array, T, dims))

# broadcast
Base.Broadcast.BroadcastStyle(::Type{<:Field}) = Broadcast.ArrayStyle{Field}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{Field}},
                      ::Type{ElType}) where ElType
  u = find_fld(bc)
  Field(similar(Array{ElType}, axes(u.array)))
end

find_fld(bc::Base.Broadcast.Broadcasted) = find_fld(bc.args)
find_fld(args::Tuple) = find_fld(find_fld(args[1]), Base.tail(args))
find_fld(x) = x
find_fld(::Tuple{}) = nothing
find_fld(a::Field, rest) = a
find_fld(::Any, rest) = find_fld(rest)

# math overloads predefined since Field <: AbstractArray

#
