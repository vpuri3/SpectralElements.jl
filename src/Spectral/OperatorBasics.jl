#
""" Common Operator Interface """

# fallbacks
function (A::AbstractOperator{Ta,D})(u::AbstractArray) where{Ta,D}
    if issquare(A)
        mul!(similar(u),A,u)
    else
        ArgumentError("Operator application not defined for $A")
    end
end

Base.:*(A::AbstractOperator{Ta,D}, u::AbstractField{Tu,D}) where{Ta,Tu,D} = A(u)

function Base.:*(A::AbstractOperator, B::AbstractOperator)
    @warn "Fusing operation not defined for $A * $B. falling back to lazy ∘"
    A ∘ B
end

# caching
function init_cache(A::AbstractOperator,u)
    @error "Caching behaviour not defined for $A"
end

function set_cache(A::AbstractOperator, cache)
    @set! A.cache = cache
    @set! A.isunset = false
    return A
end

Base.size(t::AbstractOperator, d) where {T,D} = d::Integer <= 2 ? size(t)[d] : 1

function Base.:+(A::AbstractOperator{Ta,D},
                B::AbstractOperator{Tb,D},
               ) where{Ta,Tb,D}
    AffineOperator(A,B,true,true)
end

function Base.:-(A::AbstractOperator{Ta,D},
                B::AbstractOperator{Tb,D}
               ) where{Ta,Tb,D}
    AffineOperator(A,B,true,-true)
end

function Base.:+(A::AbstractOperator{T,D}, λ::Number) where{T,D}
    Id = IdentityOp(size(A)...)
    AffineOperator(A, Id, true, λ)
end

function Base.:-(A::AbstractOperator{T,D}, λ::Number) where{T,D}
    Id = IdentityOp(size(A)...)
    AffineOperator(A, Id, -true, λ)
end

function Base.:*(A::AbstractOperator{T,D}, λ::Number) where{T,D}
    Z = ZeroOp(size(A)...)
    AffineOperator(A, Z, true, λ)
end

function Base.:/(A::AbstractOperator{T,D}, λ::Number) where{T,D}
    Z = ZeroOp(size(A)...)
    AffineOperator(A, Z, -true, λ)
end

""" Zero Operator with a notion of size """
struct ZeroOp{D,Tn} <: AbstractOperator{Bool, D}
    n::Tn # tuple of sizes
end

""" Identity Operator """
struct IdentityOp{D,Tn} <: AbstractOperator{Bool, D}
    n::Tn # tuple of sizes
    #
    function IdentityOp(n...)
        D = length(n)
        new{D,typeof(n)}(n)
    end
end
function Base.size(Id::IdentityOp)
    n = prod(Id.n)
    (n,n)
end
Base.adjoint(Id::IdentityOp) = Id

function LinearAlgebra.mul!(v::AbstractField{Tv,D},
                            Id::IdentityOp{D},
                            u::AbstractField{Tu,D}
                           ) where{Tv,Tu,D}
    copy!(v, u)
end

function LinearAlgebra.ldiv!(v::AbstractField{Tv,D},
                             Id::IdentityOp{D},
                             u::AbstractField{Tu,D}
                            ) where{Tv,Tu,D}
    copy!(v, u)
end

function LinearAlgebra.ldiv!(id::IdentityOp{D}, u::AbstractField{Tu,D}) where{Tu,D}
    u
end

Base.:*(::IdentityOp{D}, A::AbstractOperator{T,D}) where{D,T} = A
Base.:*(A::AbstractOperator{T,D}, ::IdentityOp{D}) where{D,T} = A

"""
ToArrayOp
use RecursiveArrayTools.jl: ArrayPartition, ComponentArrays.jl instead
"""
struct ToArrayOp{D,Tn} <: AbstractOperator{Bool,D}
    n::Tn # tuple of sizes
    #
    function ToArrayOp(n...)
        D = length(n)
        new{D,typeof(n)}(n)
    end
end
Base.size(C::ToArrayOp) = (C.n,C.n)

(C::ToArrayOp)(u) = fill(u, (1,))
(C::Adjoint{Bool, ToArrayOp})(u) = first(u)
LinearAlgebra.mul!(v, C::ToArrayOp, u) = copy!(first(v),u)
LinearAlgebra.ldiv!(v, C::ToArrayOp, u) = first(u)

""" Lazy Composite (affine) Operator """
struct AffineOperator{T,D,Ta,Tb} <: AbstractOperator{T,D} #where{Ta,Tb}
    A::Ta
    B::Tb
    α::T
    β::T

    function AffineOperator(A::AbstractOperator{Ta,D},
                            B::AbstractOperator{Tb,D},
                            α::Number, β::Number
                           ) where{Ta,Tb,D}

        T = promote_type(Ta,Tb)
        new{T,D,typeof(A),typeof(B)}(A, B, T(α), T(β))
    end
end

#Base.*
#LinearAlgebra.mul!()

""" Lazy Composition """
struct ComposeOperator{T,D,Ti,To,Tc} <: AbstractOperator{T,D}
    inner::Ti
    outer::To
    #
    cache::Tc
    isunset::Bool
    #
    function ComposeOperator(inner::AbstractOperator{Ti,D},
                             outer::AbstractOperator{To,D},
                             cache = nothing,
                             isunset::Bool = cache === nothing
                            ) where{Ti,To,D}
        @assert size(outer, 1) == size(inner, 2)
        T = promote_type(Ti, To)
        isunset = cache === nothing
        new{T,D,typeof(inner),typeof(outer),typeof(cache)}(inner, outer, cache, isunset)
    end
end

function Base.:∘(outer::AbstractOperator,
                 inner::AbstractOperator)
    ComposeOperator(inner,outer)
end

Base.size(A::ComposeOperator) = (size(A.outer, 1), size(A.inner, 2))
Base.adjoint(A::ComposeOperator) = A.inner' ∘ A.outer'
Base.inv(A::ComposeOperator) = inv(A.inner) ∘ inv(A.outer)

function init_cache(A::ComposeOperator{Ta,D}, u::Field{Tu,D}) where{Ta,Tu,D}
    cache = A.inner(u)
    return cache
end

function (A::ComposeOperator{Ta,D})(u::Field{Tu,D}) where{Ta,Tu,D}
    if A.isunset
        cache = init_cache(A, u)
        A = set_cache(A, cache)
        return outer(cache)
    end
    mul!(cache, inner, u)
    outer(cache)
end

function LinearAlgebra.mul!(v::Field{Tv,D}, A::ComposeOperator{Ta,D}, u::Field{Tu,D}) where{Tv,Ta,Tu,D}
    if A.isunset
        cache = init_cache(A, u)
        A = set_cache(A, cache)
        return mul!(v, outer, cache)
    end

    mul!(cache, inner, u)
    mul!(v, outer, cache)
end

function LinearAlgebra.ldiv!(A::ComposeOperator{Ta,D}, u::Field{Tu,D}) where{Ta,Tu,D}
    @unpack inner, outer = A

    ldiv!(inner, u)
    ldiv!(outer, u)
end

function LinearAlgebra.ldiv!(v::Field{Tv,D}, A::ComposeOperator{Ta,D}, u::Field{Tu,D}) where{Tv,Ta,Tu,D}
    @unpack inner, outer = A

    ldiv!(v, inner, u)
    ldiv!(outer, v)
end

""" InverseOperator """
struct InverseOperator{T,D,Ta} <: AbstractOperator{T,D}
    A::Ta
    #
    function InverseOperator(A::AbstractOperator{T,D}) where{T,D}
        @assert issquare(A)
        LinearAlgebra.checksquare(A)
        new{T,D,typeof(A)}(A)
    end
end

Base.inv(A::AbstractOperator) = InverseOperator(A)
Base.size(A::InverseOperator) = size(A.A)
Base.adjoint(A::InverseOperator) = inv(A.A')
LinearAlgebra.ldiv!(y, A::InverseOperator, x) = mul!(y, A.A, x)
LinearAlgebra.mul!(y, A::InverseOperator, x) = ldiv!(y, A.A, x)
#
