#
""" Common Operator Interface """

#SciMLBase.has_adjoint(::AbstractOperator) = true
SciMLBase.has_mul(::AbstractOperator) = true
SciMLBase.has_mul!(::AbstractOperator) = true

# * fallback
function Base.:*(A::AbstractOperator{Ta,D},u::AbstractField{Tu,D}) where{Ta,Tu,D}
    if issquare(A)
        mul!(similar(u),A,u)
    else
        ArgumentError("Operator application not defined for $A")
    end
end

# mul! fallback
function LinearAlgebra.mul!(v::AbstractField{Tv,D},A::AbstractOperator{Ta,D},u::AbstractField{Tu,D}) where{Tv,Ta,Tu,D}
    ArgumentError("Operator application not defined for $A")
end

# fusion fallback
function Base.:*(A::AbstractOperator, B::AbstractOperator)
    @warn "Operator fusion not defined for $A * $B. falling back to lazy composition, ∘"
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

Base.size(A::AbstractOperator{T,D}, d::Integer) where {T,D} = d <= 2 ? size(A)[d] : 1

""" (Square) Zero operator """
struct ZeroOp{D} <: AbstractOperator{Bool,D} end

Base.adjoint(Z::ZeroOp) = Z
issquare(::ZeroOp) = true
#function Base.size(Id::IdentityOp)
#    n = prod(Id.n)
#    (n,n)
#end

function LinearAlgebra.mul!(v::AbstractField{Tv,D}, ::ZeroOp{D}, u::AbstractField{Tu,D}) where{Tv,Tu,D}
    mul!(v,I, false)
end

Base.:*(Z::ZeroOp{D}, A::AbstractOperator{T,D}) where{D,T} = Z
Base.:*(A::AbstractOperator{T,D}, Z::ZeroOp{D}) where{D,T} = Z

Base.:∘(Z::ZeroOp{D}, A::AbstractOperator{T,D}) where{D,T} = Z
Base.:∘(A::AbstractOperator{T,D}, Z::ZeroOp{D}) where{D,T} = Z

""" (Square) Identity operator """
struct IdentityOp{D} <: AbstractOperator{Bool,D} end

Base.adjoint(Id::IdentityOp) = Id
issquare(::IdentityOp) = true
#function Base.size(Id::IdentityOp)
#    n = prod(Id.n)
#    (n,n)
#end

function LinearAlgebra.mul!(v::AbstractField{Tv,D}, ::IdentityOp{D}, u::AbstractField{Tu,D}) where{Tv,Tu,D}
    copy!(v, u)
end

function LinearAlgebra.ldiv!(v::AbstractField{Tv,D}, ::IdentityOp{D}, u::AbstractField{Tu,D}) where{Tv,Tu,D}
    copy!(v, u)
end

function LinearAlgebra.ldiv!(Id::IdentityOp{D}, u::AbstractField{Tu,D}) where{Tu,D}
    u
end

# fusion
Base.:*(::IdentityOp{D}, A::AbstractOperator{T,D}) where{D,T} = A
Base.:*(A::AbstractOperator{T,D}, ::IdentityOp{D}) where{D,T} = A

# lazy composition
Base.:∘(::IdentityOp{D}, A::AbstractOperator{T,D}) where{D,T} = A
Base.:∘(A::AbstractOperator{T,D}, ::IdentityOp{D}) where{D,T} = A

""" Lazy Combination (affine) Operator αA + βB """
struct AffineOperator{T,D,Ta,Tb,Tα,Tβ,Tc} <: AbstractOperator{T,D}
    A::Ta
    B::Tb
    α::Tα
    β::Tβ

    cache::Tc
    isunset::Bool

    function AffineOperator(A::AbstractOperator{Ta,D}, B::AbstractOperator{Tb,D}, α::Number, β::Number,
                      cache = nothing, isunset = cache === nothing) where{Ta,Tb,D}
        T = promote_type(Ta,Tb)
        new{T,D,typeof(A),typeof(B),typeof(α),typeof(β),typeof(C)}(A, B, α, β, cache, isunset)
    end
end

function Base.adjoint(A::AffineOperator)
    if issquare(A)
        AffineOperator(A.A',A.B',A.α, A.β, A.cache, A.isunset)
    else
        AffineOperator(A.A',A.B',A.α, A.β)
    end
end

function init_cache(A::AffineOperator{T,D}, u::AbstractField{T,D}) where{T,D}
    cache = A.B * u
end

function LinearAlgebra.mul!(v::AbstractField{Tv,D}, A::AffineOperator{Ta,D}, u::AbstractField{Tu,D}) where{Tv,Ta,Tu,D}
    mul!(v, A.A, u)
    lmul!(A.α, v)

    if isunset
        cache = init_cache(A,u)
        A = set_cache(A, cache)
    else
        mul!(A.cache, A.B, u)
    end

    lmul!(A.β, A.cache)
    axpy!(true, A.cache, v)
end

function Base.:*(A::AffineOperator{Ta,D}, u::AbstractField{Tu,D}) where{Ta,Tu,D}
    @unpack A, B, α, β = A
    α * (A * u) + β * (B * u)
end

function Base.:+(A::AbstractOperator{Ta,D}, B::AbstractOperator{Tb,D},) where{Ta,Tb,D}
    AffineOperator(A,B,true,true)
end

function Base.:-(A::AbstractOperator{Ta,D}, B::AbstractOperator{Tb,D}) where{Ta,Tb,D}
    AffineOperator(A,B,true,-true)
end

function Base.:+(A::AbstractOperator{T,D}, λ::Number) where{T,D}
    Id = IdentityOp{D}()
    AffineOperator(A, Id, true, λ)
end

function Base.:-(A::AbstractOperator{T,D}, λ::Number) where{T,D}
    Id = IdentityOp{D}()
    AffineOperator(A, Id, -true, λ)
end

function Base.:*(A::AbstractOperator{T,D}, λ::Number) where{T,D}
    Z = ZeroOp{D}()
    AffineOperator(A, Z, true, λ)
end

function Base.:/(A::AbstractOperator{T,D}, λ::Number) where{T,D}
    Z = ZeroOp{D}()
    AffineOperator(A, Z, -true, λ)
end

"""
Do array reductions

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
#       @assert size(outer, 1) == size(inner, 2)
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

function init_cache(A::ComposeOperator{Ta,D}, u::AbstractField{Tu,D}) where{Ta,Tu,D}
    cache = A.inner(u)
    return cache
end

function (A::ComposeOperator{Ta,D})(u::AbstractField{Tu,D}) where{Ta,Tu,D}
    if A.isunset
        cache = init_cache(A, u)
        A = set_cache(A, cache)
        return outer(cache)
    end
    mul!(cache, inner, u)
    outer(cache)
end

function LinearAlgebra.mul!(v::AbstractField{Tv,D}, A::ComposeOperator{Ta,D}, u::AbstractField{Tu,D}) where{Tv,Ta,Tu,D}
    if A.isunset
        cache = init_cache(A, u)
        A = set_cache(A, cache)
        return mul!(v, outer, cache)
    end

    mul!(cache, inner, u)
    mul!(v, outer, cache)
end

function LinearAlgebra.ldiv!(A::ComposeOperator{Ta,D}, u::AbstractField{Tu,D}) where{Ta,Tu,D}
    @unpack inner, outer = A

    ldiv!(inner, u)
    ldiv!(outer, u)
end

function LinearAlgebra.ldiv!(v::AbstractField{Tv,D}, A::ComposeOperator{Ta,D}, u::AbstractField{Tu,D}) where{Tv,Ta,Tu,D}
    @unpack inner, outer = A

    ldiv!(v, inner, u)
    ldiv!(outer, v)
end

""" InverseOperator """
struct InverseOperator{T,D,Ta} <: AbstractOperator{T,D}
    A::Ta
    #
    function InverseOperator(A::AbstractOperator{T,D}) where{T,D}
#       @assert issquare(A)
        new{T,D,typeof(A)}(A)
    end
end

Base.inv(A::AbstractOperator) = InverseOperator(A)
Base.size(A::InverseOperator) = size(A.A)
Base.adjoint(A::InverseOperator) = inv(A.A')
LinearAlgebra.ldiv!(y, A::InverseOperator, x) = mul!(y, A.A, x)
LinearAlgebra.mul!(y, A::InverseOperator, x) = ldiv!(y, A.A, x)
#
