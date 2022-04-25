#
###
# Lagrange polynomial function spaces
###

function LagrangePolySpace1D(domain::AbstractDomain{<:Number,1},
                             n = 16;
                             quadrature = gausslobatto,
                             T = Float64,
                            )
    z, w = quadrature(n)

    D = lagrange_poly_deriv_mat(z)

    z = T.(z) |> Field
    w = T.(w) |> Field

    grid = (z,)
    mass = DiagonalOp(w)
    grad = D |> MatrixOp

    gather_scatter(n, domain)

    space = SpectralSpace(grid, mass, grad, gather_scatter)
    space = domain.mapping === nothing ? space : deform(space)
end

GaussLobattoLegendre1D(args...; kwargs...) = LagrangePolySpace1D(args...; quadrature=gausslobatto, kwargs...)
GaussLegendre1D(args...; kwargs...) = LagrangePolySpace1D(args...; quadrature=gausslegendre, kwargs...)
GaussChebychev1D(args...; kwargs...) = LagrangePolySpace1D(args...; quadrature=gausschebyshev, kwargs...)

function LagrangePolySpace2D(domain::AbstractDomain{<:Number,2} nr, ns;
                             quadrature = gausslobatto,
                             T = Float64,
                            )
    zr,wr = quadrature(nr)
    zs,ws = quadrature(ns)
    
    zr,wr = T.(zr), T.(wr)
    zs,ws = T.(zs), T.(ws)

    x, y = ndgrid(zr,zs)
    grid = (x, y,)
    
    Dr = lagrange_poly_deriv_mat(zr)
    Ds = lagrange_poly_deriv_mat(zs)
    
    Ir = Matrix(I, nr, nr) |> sparse
    Is = Matrix(I, ns, ns) |> sparse

    DrOp = TensorProductOp2D(Dr,Is)
    DsOp = TensorProductOp2D(Ir,Ds)
    
    mass = w * w' |> Field |> DiagonalOp

    grad = [DrOp
            DsOp]

    space = SpectralSpace(grid, mass, grad, gather_scatter)
    space = domain.mapping === nothing ? space : deform(space)
end

GaussLobattoLegendre2D(args...; kwargs...) = LagrangePolySpace2D(args...; quadrature=gausslobatto, kwargs...)
GaussLegendre2D(args...; kwargs...) = LagrangePolySpace2D(args...; quadrature=gausslegendre, kwargs...)
GaussChebychev2D(args...; kwargs...) = LagrangePolySpace2D(args...; quadrature=gausschebyshev, kwargs...)

###
# BC implementation
###

struct Mask{D} <: AbstractBoundnaryCondition{Bool, D}
    tags
    dirichlet_func!
    neumann_func!
    mask
end


struct BC{T,D} <: AbstractBoundaryCondition{T,D}
    tags # dirichlet, neumann
    dirichlet_func! # (ub, space) -> mul!(ub, I, false)
    neumann_func!
    mask # implementation
end

struct GS{T,D} <: AbstractGatherScatterOperator{T,D}
    gsOp
    l2g  # local-to-global numbering
    g2l  # global-to-local numbering
end

""" Interpolation operator between spaces """
struct Interp2D{T,Td1,Td2} <: AbstractOperator{T,2}
    space1::Ts1 # or domain1/2?
    space2::Ts2

    # need general purpose implementation
    # need guarantees that domains match

    function Interp2D(space1, space2) where{Tx,Ty}
        T = promote_type(Tx,Ty)
        new{T,typeof(x)}(Tx,Ty)
    end
end
#
