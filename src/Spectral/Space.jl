#
###
# AbstractSpace interface
###

"""
args:
    space::AbstractSpace{T,D}
ret:
    (x1, ..., xD,) # incl end points
"""
function grid end

"""
Gradient Operator
Compute gradient of u∈H¹(Ω).

Continuity isn't enforced across
element boundaries for gradients

args:
    space::AbstractSpace{T,D}
ret:
    gradOp: u -> [dudx1, ..., dudxD]
"""
function gradOp end

"""
Mass Operator

args:
    space::AbstractSpace{T,D}
    space_dealias
ret:
    massOp: AbstractField -> AbstractField
"""
function massOp end

"""
Laplace Operator

for v,u in H¹₀(Ω)

(v,-∇² u) = (vx,ux) + (vy,uy)\n
         := a(v,u)\n

args:
    space::AbstractSpace{T,D}
    space_dealias
ret:
    laplaceOp: AbstractField -> AbstractField
"""
function laplaceOp(space::AbstractSpace)
    D = gradOp(space)
    M = massOp(space)

    lapl = D' ∘ [M] ∘ D
    first(lapl)
end

"""
args:
    space::AbstractSpace{<:Number,D}
    vel...::AbstractField{<:Number,D}
    space_dealias
ret:
    advectionOp: AbstractField -> AbstractField

for v,u,T in H¹₀(Ω)

(v,(u⃗⋅∇)T) = (v,ux*∂xT + uy*∂yT)\n
           = v' *B*(ux*∂xT + uy*∂yT)\n

implemented as

R'R * QQ' * B * (ux*∂xT + uy*∂yT)

(u⃗⋅∇)T = ux*∂xT + uy*∂yT

       = [ux uy] * [Dx] T
                   [Dx]

ux,uy, ∇T are interpolated to
a grid with higher polynomial order
for dealiasing (over-integration)
so we don't commit any
"variational crimes"

"""
function advectionOp(space::AbstractSpace{<:Number,D},
                     vel::AbstractField{<:Number,D}...
                    ) where{D}
    V = [DiagonalOp.(vel)...]

    D = gradOp(space)
    M = massOp(space)

    advectOp = V' * [M] * D

    first(advectOp)
end

###
# Dealiased operators
###

function massOp(space1::AbstractSpace{<:Number,D},
                space2::AbstractSpace{<:Number,D},
                interpOp = nothing
               ) where{D}
    J12 = interpOp !== nothing ? J : interpOp(space1, space2)

    M2 = massOp(space2)

    J12' ∘ M2 ∘ J12
end

function laplaceOp(space1::AbstractSpace{<:Number,D},
                   space2::AbstractSpace{<:Number,D},
                   J = nothing
                  ) where{D}
    J12 = J !== nothing ? J : interpOp(space1, space2)

    M2 = massOp(space2)
    D1 = gradOp(space1)
    JD = [J12] .∘ D1

    laplOp = JD' ∘ [M2] ∘ JD

    first(laplOp)
end

function advectionOp(space1::AbstractSpace{<:Number,D},
                     space2::AbstractSpace{<:Number,D},
                     vel::AbstractField{<:Number,D}...,
                     J = nothing
                    ) where{d}
    J12 = interpOp !== nothing ? J : interpOp(space1, space2)

    V1 = [DiagonalOp.(vel)...]
    V2 = J12 .* V1

    M2 = massOp(space2)
    D1 = gradOp(space1)

    advectOp = [J12]' .∘ V2' .∘ [M2] .∘ [J12] .∘ D1

    first(advectOp)
end

###
# Tensor Product Spaces
###

struct TensorProductSpace{T} <: AbstractTensorProductSpace{T,D}
    space1
    space2
end
#
