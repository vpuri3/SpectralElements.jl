#
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

    xr = DrOp(x)
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
struct Gradient2D{T} <: AbstractSpectralOperator{T,2}
  grad
  #
  function Gradient2D(space::AbstractSpectralSpace{T,2})

    ddr = [DrOp
           DsOp] |> hcat

    grad = @. dRdX ∘ ddr

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

""" Convection Operator """
struct Convection{T,N,fldT}
    v::fldT
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
function apply_bc(u::Field, bc::BC)
  # apply mask
  # add dirichlet data (?)
end

""" Gather-Scatter Operator """
struct GatherScatter{T,N} # periodic condition, elemenet-wise GS
    l2g
    g2l
end

export SpectralSpace2D
""" Tensor Product Polynomial Spectral Space """
struct SpectralSpace2D{T,vecT,fldT,massT,derivT,interpT,funcT} <: AbstractSpectralSpace{T,2}
    nr::Int
    ns::Int

    zr::vecT
    wr::vecT

    zs::vecT
    ws::vecT

    r::fldT  # reference coordinates
    s::fldT

    x::fldT  # spatial coordinates
    y::fldT

    mass  ::massT 
    deriv ::derivT
    interp::interpT

    deform::funcT
    #
    function SpectralSpace2D(nr::Int = 8, ns::Int = 8, T=Float64;
                             quadrature = FastGaussQuadrature.gausslobatto,
                             deform::Function = (r,s) -> (copy(r), copy(s)),
                            )
        zr,wr = quadrature(nr)
        zs,ws = quadrature(ns)
    
        zr,wr = T.(zr), T.(wr)
        zs,ws = T.(zs), T.(ws)
    
        o = ones(T,n)
        r = z * o' |> Field
        s = o * z' |> Field
    
        x,y = deform(r,s)
    
        B  = w * w'
        Dr = derivMat(zr)
        Ds = derivMat(zs)

        Ir = Identity(nr)
        Is = Identity(ns)

        DrOp = TensorProduct2D(Dr,Is)
        DsOp = TensorProduct2D(Ir,Ds)

        DrDsOp = [DrOp
                  DsOp] |> hcat

#       jac  = 
#       grad =
#       mass = B * jac
#       dealias2 = 
    
        ifperiodic = [false,false]
    
        return new{T}(nr,nsz,w,r,s,x,y,B,D, deform)
    end
end
size(space::SpectralSpace2D) = (space.nr * space.ns,)

GaussLobattoLegendre2D(args...;kwargs...) = SpectralSpace2D(args...; quadrature=FastGaussQuadrature.gausslobatto, kwargs...)
GaussLegendre2D(args...;kwargs...) = SpectralSpace2D(args...; quadrature=FastGaussQuadrature.gausslegendre, kwargs...)
GaussChebychev2D(args...;kwargs...) = SpectralSpace2D(args...; quadrature=FastGaussQuadrature.gausschebyshev, kwargs...)
#
