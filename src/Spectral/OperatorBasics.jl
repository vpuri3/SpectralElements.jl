#
###
# Traits and fallbacks
###

SciMLBase.has_adjoint(::AbstractOperator) = true
SciMLBase.has_mul(::AbstractOperator) = true
SciMLBase.has_mul!(::AbstractOperator) = true

function Base.:*(A::AbstractOperator{<:Number,D},u::AbstractField{<:Number,D}) where{D}
    if issquare(A)
        mul!(similar(u),A,u)
    else
        ArgumentError("Operator application not defined for $A")
    end
end

function LinearAlgebra.mul!(v::AbstractField{<:Number,D},A::AbstractOperator{<:Number,D},u::AbstractField{<:Number,D}) where{D}
    ArgumentError("LinearAlgebra.mul! not defined for $A")
end

function Base.:\(A::AbstractOperator{<:Number,D},u::AbstractField{<:Number,D}) where{D}
    ArgumentError("Operator inversion not defined for $A")
end

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

Base.size(A::AbstractOperator, d::Integer) = d <= 2 ? size(A)[d] : 1

###
# ZeroOp
###

""" (Square) Zero operator """
struct ZeroOp{D} <: AbstractOperator{Bool,D} end

Base.adjoint(Z::ZeroOp) = Z
issquare(::ZeroOp) = true

function LinearAlgebra.mul!(v::AbstractField{<:Number,D}, ::ZeroOp{D}, u::AbstractField{<:Number,D}) where{D}
    mul!(v,I, false)
end

# overload fusion
Base.:*(Z::ZeroOp{D}, A::AbstractOperator{<:Number,D}) where{D} = Z
Base.:*(A::AbstractOperator{<:Number,D}, Z::ZeroOp{D}) where{D} = Z

# overload composition
Base.:∘(Z::ZeroOp{D}, A::AbstractOperator{<:Number,D}) where{D} = Z
Base.:∘(A::AbstractOperator{<:Number,D}, Z::ZeroOp{D}) where{D} = Z

###
# IdentityOp
###

""" (Square) Identity operator """
struct IdentityOp{D} <: AbstractOperator{Bool,D} end

SciMLBase.has_ldiv(::IdentityOp) = true
SciMLBase.has_ldiv!(::IdentityOp) = true

Base.adjoint(Id::IdentityOp) = Id
issquare(::IdentityOp) = true

function LinearAlgebra.mul!(v::AbstractField{<:Number,D}, ::IdentityOp{D}, u::AbstractField{<:Number,D}) where{D}
    copy!(v, u)
end

function LinearAlgebra.ldiv!(v::AbstractField{<:Number,D}, ::IdentityOp{D}, u::AbstractField{<:Number,D}) where{D}
    copy!(v, u)
end

function LinearAlgebra.ldiv!(Id::IdentityOp{D}, u::AbstractField{<:Number,D}) where{D}
    u
end

# overload fusion
Base.:*(::IdentityOp{D}, A::AbstractOperator{<:Number,D}) where{D} = A
Base.:*(A::AbstractOperator{<:Number,D}, ::IdentityOp{D}) where{D} = A

# overload composition
Base.:∘(::IdentityOp{D}, A::AbstractOperator{<:Number,D}) where{D} = A
Base.:∘(A::AbstractOperator{<:Number,D}, ::IdentityOp{D}) where{D} = A

###
# AffineOp
###

""" Lazy affine operator combinations αA + βB """
struct AffineOp{T,D,
                Ta <: AbstractOperator{<:Number,D},
                Tb <: AbstractOperator{<:Number,D},
                Tα,Tβ,Tc
               } <: AbstractOperator{T,D}
    A::Ta
    B::Tb
    α::Tα
    β::Tβ

    cache::Tc
    isunset::Bool

    function AffineOp(A::AbstractOperator{Ta,D}, B::AbstractOperator{Tb,D}, α, β,
                      cache = nothing, isunset = cache === nothing) where{Ta,Tb,D}
        T = promote_type(Ta,Tb)
        new{T,D,typeof(A),typeof(B),typeof(α),typeof(β),typeof(C)}(A, B, α, β, cache, isunset)
    end
end

issquare(A::AffineOp) = issquare(A.A) & issquare(A.B)
function Base.adjoint(A::AffineOp)
    if issquare(A)
        AffineOp(A.A',A.B',A.α, A.β, A.cache, A.isunset)
    else
        AffineOp(A.A',A.B',A.α, A.β)
    end
end

function init_cache(A::AffineOp{<:Number,D}, u::AbstractField{<:Number,D}) where{D}
    cache = A.B * u
end

function Base.:*(A::AffineOp{<:Number,D}, u::AbstractField{<:Number,D}) where{D}
    @unpack A, B, α, β = A
    α * (A * u) + β * (B * u)
end

function LinearAlgebra.mul!(v::AbstractField{<:Number,D}, A::AffineOp{<:Number,D}, u::AbstractField{<:Number,D}) where{D}
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

function Base.:+(A::AbstractOperator{<:Number,D}, B::AbstractOperator{<:Number,D},) where{D}
    AffineOp(A,B,true,true)
end

function Base.:-(A::AbstractOperator{<:Number,D}, B::AbstractOperator{<:Number,D}) where{D}
    AffineOp(A,B,true,-true)
end

function Base.:+(A::AbstractOperator{<:Number,D}, λ::Number) where{D}
    Id = IdentityOp{D}()
    AffineOp(A, Id, true, λ)
end

function Base.:-(A::AbstractOperator{<:Number,D}, λ::Number) where{D}
    Id = IdentityOp{D}()
    AffineOp(A, Id, -true, λ)
end

function Base.:*(A::AbstractOperator{<:Number,D}, λ::Number) where{D}
    Z = ZeroOp{D}()
    AffineOp(A, Z, true, λ)
end

function Base.:/(A::AbstractOperator{<:Number,D}, λ::Number) where{D}
    Z = ZeroOp{D}()
    AffineOp(A, Z, -true, λ)
end

###
# ComposeOp
###

""" Lazy Composition A ∘ B """
struct ComposeOp{T,D,Ti,To,Tc} <: AbstractOperator{T,D}
    inner::Ti
    outer::To

    cache::Tc
    isunset::Bool

    function ComposeOp(inner::AbstractOperator{Ti,D},
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

function Base.:∘(outer::AbstractOperator, inner::AbstractOperator)
    ComposeOp(inner,outer)
end

Base.size(A::ComposeOp) = (size(A.outer, 1), size(A.inner, 2))
Base.adjoint(A::ComposeOp) = A.inner' ∘ A.outer'
Base.inv(A::ComposeOp) = inv(A.inner) ∘ inv(A.outer)

SciMLBase.has_ldiv(A::ComposeOp) = has_ldiv(A.inner) & has_ldiv(A.outer)
SciMLBase.has_ldiv!(A::ComposeOp) = has_ldiv!(A.inner) & has_ldiv!(A.outer)
issquare(A::ComposeOp) = issquare(A.inner) & issquare(A.outer)

function init_cache(A::ComposeOp{<:Number,D}, u::AbstractField{<:Number,D}) where{D}
    cache = A.inner * u
    return cache
end

function Base.:*(A::ComposeOp{<:Number,D}, u::AbstractField{<:Number,D}) where{D}
    @unpack inner, outer = A
    outer * inner * u
end

function Base.:\(A::ComposeOp{<:Number,D}, u::AbstractField{<:Number,D}) where{D}
    @assert has_ldiv(A)
    @unpack inner, outer = A

    outer \ (inner \ u)
end

function LinearAlgebra.mul!(v::AbstractField{<:Number,D}, A::ComposeOp{<:Number,D}, u::AbstractField{<:Number,D}) where{D}
    @unpack inner, outer = A

    if A.isunset
        cache = init_cache(A, u)
        A = set_cache(A, cache) # pass this A to calling context
    else
        mul!(cache, inner, u)
    end

    mul!(v, outer, cache)
end

function LinearAlgebra.ldiv!(A::ComposeOp{<:Number,D}, u::AbstractField{<:Number,D}) where{D}
    @assert has_ldiv!(A)
    @unpack inner, outer = A

    ldiv!(inner, u)
    ldiv!(outer, u)
end

function LinearAlgebra.ldiv!(v::AbstractField{<:Number,D}, A::ComposeOp{<:Number,D}, u::AbstractField{<:Number,D}) where{D}
    @assert has_ldiv!(A)
    @unpack inner, outer = A

    ldiv!(v, inner, u)
    ldiv!(outer, v)
end

###
# InverseOp
###

""" Lazy Inverse Operator """
struct InverseOp{T,D,Ta} <: AbstractOperator{T,D}
    A::Ta

    function InverseOp(A::AbstractOperator{T,D}) where{T,D}
        @assert issquare(A)
        new{T,D,typeof(A)}(A)
    end
end

function Base.:*(A::InverseOp{<:Number,D}, u::AbstractField{<:Number,D}) where{D}
    A.A \ u
end

SciMLBase.has_ldiv(::InverseOp) = true
SciMLBase.has_ldiv!(::InverseOp) = true
issquare(::InverseOp) = true

Base.inv(A::AbstractOperator) = InverseOp(A)
Base.size(A::InverseOp) = size(A.A)
Base.adjoint(A::InverseOp) = inv(A.A')
LinearAlgebra.ldiv!(y, A::InverseOp, x) = mul!(y, A.A, x)
LinearAlgebra.mul!(y, A::InverseOp, x) = ldiv!(y, A.A, x)
#
