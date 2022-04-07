#
include("NDgrid.jl")
include("DerivMat.jl")
include("InterpMat.jl")

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
struct Deformation2D{T,N,Tjac,fldT}
  deform
  J ::Tjac
  Ji::Tjac
  dXdR
  dRdX
  #
  function Deformation2D(deform,r,s,DrOp,DsOp)
    x,y = deform(r,s)

    xr = DrOp(x) # fields
    xs = DsOp(x)
    yr = DrOp(y)
    ys = DsOp(y)
        
    dXdR = DiagonalOp.([xr xs
                        yr ys])

    J  = @. xr * ys - xs * yr
    Ji = @. 1 / J

    rx = @.  Ji * ys
    ry = @. -Ji * xs
    sx = @. -Ji * yr
    sy = @.  Ji * xr

    dRdX = DiagonalOp.[rx sx
                       ry sy]

    new{T}()
  end
end

""" mass """
struct Mass2D{T}
  B
  #
  function Mass2D(B,Bd)
    mass  = DiagonalOp(B)
    massd = DiagonalOp(Bd)

    InterpVD = TensorProductOp(JrVD,JsVD)
    mass_dealias = InterpVD' ∘ massd ∘ InterpVD

    new{T}()
  end
end

"""
 Gradient Operator
 Compute gradient of u∈H¹(Ω).

 Continuity isn't enforced across
 element boundaries for gradients

 [Dx] * u = [rx sx] * [Dr] * u
 [Dy]     = [ry sy]   [Ds]

"""
struct Gradient2D{T}
  grad
  #
  function Gradient2D(space::AbstractSpace{T,2})

    ddR = [DrOp
          DsOp] |> hcat

    grad = @. dRdX ∘ ddR # == ddX

    new{T,}()
  end
end

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

"""
struct LaplaceOp2D{T,N,fldT}
  G::Matrix{fldT}

  function LaplaceOp2D
    G11 = B * (rx * rx + ry * ry)
    G12 = B * (rx * sx + ry * sy)
    G22 = B * (sx * sx + sy * sy)

    G = [G11 G12
         G12 G22]

    LaplaceOp = @. GradOp' ∘ G ∘ GradOp
    new{T}()
  end
end

"""
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
struct ConvectionOp{T,fldT} <: AbstractOperator{T,2}
  v::fldT
  C::

  function ConvectionOp(space::TensorProduct2DSpace, v::Field{T,N})
    ∘GradOp

    new{}()
  end
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

export TensorProduct2DSpace
""" Tensor Product Polynomial Space """
struct TensorProduct2DSpace{T,Tcoords,} <: AbstractSpace{T,2}
  domain_ref
  domain_phys
  domain_dealais

  mass::Tmass
  grad::Tgrad
  interp::Tinterp

  #
  function TensorProduct2DSpace(nr::Int = 8, ns::Int = 8, T=Float64;
                                quadrature = FastGaussQuadrature.gausslobatto,
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

    Ir = Identity(nr)
    Is = Identity(ns)

    DrOp = TensorProduct2D(Dr,Is)
    DsOp = TensorProduct2D(Ir,Ds)

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
Base.size(space::TensorProduct2DSpace) = (space.nr * space.ns,)

GaussLobattoLegendre2D(args...;kwargs...) = TensorProduct2DSpace(args...; quadrature=FastGaussQuadrature.gausslobatto, kwargs...)
GaussLegendre2D(args...;kwargs...) = TensorProduct2DSpace(args...; quadrature=FastGaussQuadrature.gausslegendre, kwargs...)
GaussChebychev2D(args...;kwargs...) = TensorProduct2DSpace(args...; quadrature=FastGaussQuadrature.gausschebyshev, kwargs...)
#
