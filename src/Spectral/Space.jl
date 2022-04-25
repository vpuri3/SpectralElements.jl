#
""" D-Dimensional tensor-product space """
abstract type AbstractTensorSpace{T,D} <: AbstractSpace{T,D} end

""" Gather-Scatter operator in D-Dimensional space """
abstract type AbstractGatherScatterOperator{T,D} <: AbstractOperator{T,D} end

""" Interpolation operator between D-Dimensional spaces """
abstract type AbstractInterpolationOperator{T,D} <: AbstractOperator{T,D} end

"""
args:
    space::AbstractSpace{T,D}
ret:
    (x1, ..., xD,)
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
          = v' * A * u\n
          = (Q*R'*v)'*A_l*(Q*R'*u)\n
          = v'*R*Q'*A_l*Q*R'*u\n

implemented as

R'R * QQ' * A_l * u_loc
where A_l is

[Dr]'*[rx sx]'*[B 0]*[rx sx]*[Dr]
[Ds]  [ry sy]  [0 B] [ry sy] [Ds]

= [Dr]' * [G11 G12]' * [Dr]
  [Ds]    [G12 G22]    [Ds]

args:
    space::AbstractSpace{T,D}
    space_dealias
ret:
    laplaceOp: AbstractField -> AbstractField
"""
function laplaceOp end

"""
args:
    space::AbstractSpace{<:Number,D}
    vel...::AbstractField{<:Number,D}
    space_dealias
ret:
    advectOp: AbstractField -> AbstractField

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
function advectOp end

"""
AbstractSpace{T,D}
    grid # (x1, ..., xD,)

    inner_product # overload *(Adjoint{Field}, Field), norm(Field, 2)
end
"""
struct SpectralSpace{T,D,Td,Tg,Tmass,Tgrad,Tgs} <: AbstractSpectralSpace{T,D}

    domain::Td # assert [-1,1]^d, mapping == nothing
    grid::Tg   # (x1, ..., xD) # including end points
    mass::Tmass
    grad::Tgrad
    gs::Tgs
#   inner_product::Tipr # needed for SpectralElement
    # just overload *(Adjoint{Field}, Field), norm(Field, 2)
end

function grid(space::SpectralSpace)
    space.grid
end

function massOp(space::SpectralSpace)
    space.mass
end

function gradOp(space::SpectralSpace)
    space.grad
end

function laplaceOp(space::SpectralSpace)
    grad = gradOp(space)
    mass = massOp(space)

    lapl = grad' ∘ [mass] ∘ grad
    first(lapl)
end

function advectOp(space::AbstractSpace{<:Number,D},
                  vel::AbstractField{<:Number,D}...
                 ) where{D}
    V = [DiagonalOp.(vel)...]

    grad = gradOp(space)
    mass = massOp(space)

    advectOp = V' * [mass] * grad

    return first(advectOp)
end

struct TensorSpace{T,D} <: AbstractTensorSpace{T,D}
    spaces

    mass
    grad

    function TensorSpace(spaces::AbstractTensorSpace...)
        space
    end
end

struct TensorSpace2D{T} <: AbstractTensorSpace{T,2}
    space1
    space2

    mass
    grad
end
#
