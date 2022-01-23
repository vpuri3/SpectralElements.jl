#
export SpectralSpace
""" Tensor Product Polynomial Spectral Space """
struct SpectralSpace{T,vecT,fldT,massT,derivT,interpT,funcT} <: AbstractSpectralSpace{T,2}
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
end

function SpectralSpace2D(nr::Int = 8, ns::Int = 8, T=Float64;
                         quadrature::Function = FastGaussQuadrature.gausslobatto,
                         deform::Function = (r,s) -> (copy(r), copy(s)),
                        )
    zr,wr = quadrature(nr)
    zs,ws = quadrature(ns)

    zr,wr = T.(zr), T.(wr)
    zs,ws = T.(zs), T.(ws)

    o = T.(ones(T,n))
    r = z * o' |> Field
    s = o * z' |> Field

    x,y = deform(r,s)

    B = w * w'      |> DiagonalOp
    D = derivMat(z) |> TensorProdOp

    ifperiodic = [false,false]

    return SpectralSpace(z,w,r,s,x,y,B,D, deform)
end

GaussLobattoLegendre2D(args...;kwargs...) = SpectralSpace2D(args...; quadrature=FastGaussQuadrature.gausslobatto, kwargs...)

GaussLegendre2D(args...;kwargs...) = SpectralSpace2D(args...; quadrature=FastGaussQuadrature.gausslegendre, kwargs...)

GaussChebychev2D(args...;kwargs...) = SpectralSpace2D(args...; quadrature=FastGaussQuadrature.gausschebyshev, kwargs...)

""" Jacobian """
struct Jac{T,N,jacT,fldT}
    J ::jacT
    Ji::jacT
    rx::Matrix{fldT} # [rx sx; ry sy]
end

""" Boundary Condition - i.e. masking operator """
struct BC{T,N,Tm,Tb}
    mask::Tm
    bc::Tb
end

""" Copying Operator """
struct GatherScatter{T,N}
    l2g
    g2l
end

#include("derivMat.jl")
#include("interp.jl")

#include("mask.jl")
#include("mass.jl") # boundary operators, etc

#include("lapl.jl")
#incldue("advect")
#incldue("hlmz")

""" Gradient Operator """
struct Gradient{T,N}
end

""" Laplace Operator """
struct Laplace{T,N,fldT}
    G::Matrix{fldT}
end

""" Convection Operator """
struct Convection{T,N,fldT}
    v::fldT
end
#
