#
include("NDgrid.jl")
include("DerivMat.jl")
include("InterpMat.jl")

"""
args:
    space::AbstractSpace{T,D}
ret:
    x1, ..., xD
"""
function mesh end

"""
args:
    space::AbstractSpace{T,D}
ret:
    gradOp: u -> dudx1, ..., dudxD # array or vec?
"""
function gradOp end

"""
args:
    space::AbstractSpace{T,D}
    dealias=true
ret:
    massOp: AbstractField -> AbstractField

Gradient Operator
Compute gradient of u∈H¹(Ω).

Continuity isn't enforced across
element boundaries for gradients

[Dx] * u = [rx sx] * [Dr] * u
[Dy]     = [ry sy]   [Ds]

"""
function massOp end

"""
args:
    space::AbstractSpace{T,D}
    dealias = false
ret:
    laplOp: AbstractField -> AbstractField

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

"""
function laplOp end

"""
args:
    space::AbstractSpace{<:Number,D}
    vel...::AbstractField{<:Number,D}
    dealias = true
ret:
    convOp: AbstractField -> AbstractField

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
AbstractSpectralSpace{T,D}
    x1::AbstractField{T,D}
    ...
    xD::AbstractField{T,D}

    inner_product # overload *(Adjoint{Field}, Field), norm(Field, 2)
end
"""

struct SpectralSpace1D{
                       T,Tfield<:Vector{T},Tbc,Tgrad,Tmass,Tlapl,Tipr,Tcache
                      } <: AbstractSpectralSpace{T,1}
    ref_space   # store Dr, Ds, B, GS, etc here
    ref_domain  # r
    phys_domain # x
    deal_domain
    mapping     # J (=dX/dR), Ji (=dR/dX)
    interpPD

    Dr::Td
    GS::Tgs
    x::Tfield
    inner_product::Tipr

    massOp::Tmass
    gradOp::Tgrad
    laplOp::Tlapl
end

struct BC1D{T} <: AbstractBoundaryCondition{T,1}
    bctag # dirichlet, neumann
    dirichlet_func! # (ub, space) -> mul!(ub, I, false)
    neumann_func!
    mask # implementation
end

struct GS1D{T} <: AbstractGatherScatter{T,1}
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
    ref_space::Ts
    mesh::Tm # (x1, ..., xD)
    mapping::F
    dXdR::Tjacmat
    dRdx::Tjacimat
    J::Tjac
    Ji::Tjaci
end

function deform(space::AbstractSpace{<:Number, 2}, deform = (r,s) -> (r,s))
    r,s = mesh(space)
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

    DeformedSpace(space, mesh, )
end

function MassOp(space::DeformedSpace)
end

function LaplaceOp(space::DeformedSpace)
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
        V = Vd
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

""" Tensor Product Polynomial Space """
struct SpectralSpace2D{T,Tcoords,} <: AbstractSpace{T,2}
  domain_ref
  domain_phys
  domain_dealais

  mass::Tmass
  grad::Tgrad
  interp::Tinterp

  #
  function SpectralSpace2D(nr::Int = 8, ns::Int = 8, T=Float64;
                                quadrature = gausslobatto,
                                deform::Function = (r,s) -> (copy(r), copy(s)),
                                dealias::Bool = true
                               )
    zr,wr = quadrature(nr)
    zs,ws = quadrature(ns)
  
    zr,wr = T.(zr), T.(wr)
    zs,ws = T.(zs), T.(ws)
  
    r,s = ndgrid(zr,zs)
    x,y = deform(r,s)
  
    B  = w * w' |> Field |> DiagonalOp
    Dr = derivMat(zr)
    Ds = derivMat(zs)

    Ir = Matrix(I, nr, nr) |> sparse
    Is = Matrix(I, ns, ns) |> sparse

    DrOp = TensorProductOp2D(Dr,Is)
    DsOp = TensorProductOp2D(Ir,Ds)

    DrDsOp = [DrOp
              DsOp] |> hcat

#   jac  = 
#   grad =
#   mass = B * jac
#   dealias2 = 
  
    ifperiodic = [false,false]

    return new{T}(coords_ref,coords_def,coords_dealias,interp,mass,grad,)
  end
end
Base.size(space::SpectralSpace2D) = (space.nr * space.ns,)

GaussLobattoLegendre2D(args...;kwargs...) = SpectralSpace2D(args...; quadrature=gausslobatto, kwargs...)
GaussLegendre2D(args...;kwargs...) = SpectralSpace2D(args...; quadrature=gausslegendre, kwargs...)
GaussChebychev2D(args...;kwargs...) = SpectralSpace2D(args...; quadrature=gausschebyshev, kwargs...)
#
