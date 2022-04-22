#
include("LagrangePoly.jl")
include("NDgrid.jl")

"""
args:
    space::AbstractSpace{T,D}
ret:
    x1, ..., xD # tuple for now
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
    dealias=true
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
    dealias = false
ret:
    laplOp: AbstractField -> AbstractField
"""
function laplOp end

"""
args:
    space::AbstractSpace{<:Number,D}
    vel...::AbstractField{<:Number,D}
    dealias = true
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

#   domain
#   ref_grid
#   phys_grid

    domain::Td # assert [-1,1]^d

    grid::Tg   # (x1, ..., xD) # including end points
    mass::Tmass
    grad::Tgrad
    gs::Tgs
#   inner_product::Tipr # needed for SpectralElement, not Spectral
end

function grid(space::SpectralSpace)
    space.phys_grid
end

function massOp(space::SpectralSpace, dealias=false)
    !dealias ? space.mass : 1
end

function gradOp(space::SpectralSpace)
    space.grad
end

function laplOp(space::SpectralSpace, dealias=false)
    grad = gradOp(space, dealias)
    mass = massOp(space, dealias)

    lapl = grad' ∘ [mass] ∘ grad
    first(lapl)
end

function SpectralSpace1D(domain::AbstractDomain{<:Number,1}, n;
                         quadrature=gausslobatto, deform=nothing, T=Float64)
    z, w = quadrature(n)

    D  = derivMat(z)

    z = T.(z) |> Field
    w = T.(w) |> Field

    grid = z
    mass = DiagonalOp(w)
    grad = D |> MatrixOp

    gather_scatter(n, domain)

    SpectralSpace(grid, mass, grad, gather_scatter)
end

function SpectralSpace2D(nr::Int = 8, ns::Int = 8, T=Float64;
                         quadrature = gausslobatto,
                         deform::Function = (x,y) -> (x,y),
                         dealias::Bool = true
                        )
    zr,wr = quadrature(nr)
    zs,ws = quadrature(ns)
    
    zr,wr = T.(zr), T.(wr)
    zs,ws = T.(zs), T.(ws)
    
    x, y = ndgrid(zr,zs)
    grid = (x, y)
    
    Dr = derivMat(zr)
    Ds = derivMat(zs)
    
    Ir = Matrix(I, nr, nr) |> sparse
    Is = Matrix(I, ns, ns) |> sparse
    
    DrOp = TensorProductOp2D(Dr,Is)
    DsOp = TensorProductOp2D(Ir,Ds)
    
    massOp  = w * w' |> Field |> DiagonalOp
    
    gradOp = [DrOp
              DsOp]
    
    return SpectralSpace(grid, massOp, gradOp)
end

Base.size(space::SpectralSpace2D) = (space.nr * space.ns,)

GaussLobattoLegendre2D(args...;kwargs...) = SpectralSpace2D(args...; quadrature=gausslobatto, kwargs...)
GaussLegendre2D(args...;kwargs...) = SpectralSpace2D(args...; quadrature=gausslegendre, kwargs...)
GaussChebychev2D(args...;kwargs...) = SpectralSpace2D(args...; quadrature=gausschebyshev, kwargs...)

struct BC{T,D} <: AbstractBoundaryCondition{T,1}
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

"""
 Computes Jacobian and its inverse of transformation

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

function deform(space::AbstractSpace{<:Number, 2}, deform = (r,s) -> (r,s))
    r,s = grid(space)
    x,y = deform(r,s)

    grad = gradOp(space)

    dxdr, dxds = grad(x)
    dydr, dyds = grad(y)

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

function massOp(space::DeformedSpace, dealias=false)
end

function laplaceOp(space::DeformedSpace)
    gradR = gradOp(space.ref_space)

    mass = MassOp(space, dealias)
    dRdX = jacOp(space, dealias) # space.dRdX

    M = Diagonal([mass, mass])
    G = dRdX' * M * dRdX

    laplOp = @. gradR' .∘ G .∘ gradR

    return first(laplOp)
end

function advectOp(space::AbstractSpace{<:Number,D}, vel...::AbstractField{<:Number,D}, dealias=true) where{D}
    V = DiagonalOp.([vel])

    grad = gradOp(space)

    if dealias
        J =
        V = J * V
    end

    mass = massOp(space, dealias)
end

"""
 Boundary Condition
 mask, dirichlet/neumann data
"""
struct BC{T,N,Tb,Tm,Td}
  bc::Tb
  mask::Tm # <-- DiagonalOp
  data::Td
end

""" Gather-Scatter Operator """
struct GatherScatter{T,N} # periodic condition, elemenet-wise GS
  l2g
  g2l
end
#
