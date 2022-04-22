#

"""
args:
    space::AbstractSpace{T,D}
ret:
    (x1, ..., xD)
"""
function grid end

"""
args:
    space::AbstractSpace{T,D}
ret:
    gradOp: u -> [dudx1, ..., dudxD]
"""
function gradOp end

"""
Gradient Operator
Compute gradient of u∈H¹(Ω).

Continuity isn't enforced across
element boundaries for gradients

[Dx] * u = [rx sx] * [Dr] * u
[Dy]     = [ry sy]   [Ds]

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
    laplOp: AbstractField -> AbstractField
"""
function laplOp end

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
    grid # (x1, ..., xD)

    inner_product # overload *(Adjoint{Field}, Field), norm(Field, 2)
end
"""
struct SpectralSpace{T,D,
                    } <: AbstractSpectralSpace{T,D}

    domain::Td # assert [-1,1]^d, mapping == nothing
    grid::Tg   # (x1, ..., xD) # including end points
    mass::Tmass
    grad::Tgrad
    gs::Tgs
    inner_product::Tipr # needed for SpectralElement
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

function laplOp(space::SpectralSpace)
    grad = gradOp(space)
    mass = massOp(space)

    lapl = grad' ∘ [mass] ∘ grad
    first(lapl)
end

function Base.size(space::SpectralSpace)
    n = grid(space)[1] |> length
    (n,n)
end

"""
Deform domain, compute Jacobian of transformation, and its inverse

x = x(r,s), y = y(r,s)

J = [xr xs],  Jinv = [rx ry]
    [yr ys]          [sx sy]

[Dx] * u = [rx sx] * [Dr] * u
[Dy]     = [ry sy]   [Ds]

⟹
[1 0] = [rx sx] *  [Dr] [x y]
[0 1]   [ry sy]    [Ds]

⟹
                 -1
[rx sx] = [xr yr]
[ry sy]   [xs ys]

"""
struct DeformedSpace{T,D,Ts<:AbstractSpace{T,D},Tj,Tji} <: AbstractSpace{T,D}
    space::Ts
    grid::Tg # (x1, ..., xD)
    dXdR::Tjacmat
    dRdX::Tjacimat
    J::Tjac
    Ji::Tjaci
end

function deform(space::AbstractSpace, mapping = x -> x)
    R = grid(space)
    X = mapping(R...)

    gradR = gradOp(space)

    dXdR = gradR.(X)
    dXdR = hcat(dXdR)'
    dXdR = DiagonalOp.(dXdR)

    J  = det(dXdR)
    Ji = 1 / J

end

function deform(space::AbstractSpace{<:Number,2}, mapping = (r,s) -> (r,s))
    r, s = grid(space)
    x, y = mapping(r, s)

    grad = gradOp(space)

    dxdr, dxds = grad * x
    dydr, dyds = grad * y

    # Jacobian matrix
    dXdR = DiagonalOp.([dxdr dxds
                        dydr dyds])

    J  = det(dXdR) # dxdr * dyds - dxds * dydr
    Ji = 1 / J

    drdx =  (Ji * dyds)
    drdy = -(Ji * dxds)
    dsdx = -(Ji * dydr)
    dsdy =  (Ji * dxdr)

    dRdX = DiagonalOp.[drdx dsdx
                       drdy dsdy]

    DeformedSpace(space, (x,y), dXdR, dRdX, J, Ji)
end

function massOp(space::DeformedSpace)
end

function laplaceOp(space::DeformedSpace)
    gradR = gradOp(space.ref_space)

    mass = MassOp(space)
    dRdX = jacOp(space) # space.dRdX

    M = Diagonal([mass, mass])
    G = dRdX' * M * dRdX

    laplOp = @. gradR' .∘ G .∘ gradR

    return first(laplOp)
end

function advectOp(space::AbstractSpace{<:Number,D}, vel...::AbstractField{<:Number,D}) where{D}
    V = DiagonalOp.([vel])

    grad = gradOp(space)

    mass = massOp(space)
end

struct BC{T,D} <: AbstractBoundaryCondition{T,D}
    tags # dirichlet, neumann
    dirichlet_func! # (ub, space) -> mul!(ub, I, false)
    neumann_func!
    mask # implementation
end

struct GS{T,D} <: AbstractGatherScatter{T,D}
    gsOp
    l2g  # local-to-global
    g2l  # global-to-local
end

""" Gather-Scatter Operator """
struct GatherScatter{T,N} # periodic condition, elemenet-wise GS
  l2g
  g2l
end
#
