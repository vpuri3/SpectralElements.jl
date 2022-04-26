#
###
# Lagrange polynomial function spaces
###

function LagrangePolySpace1D(domain::AbstractDomain{<:Number,1},
                             n = 16;
                             quadrature = gausslobatto,
                             T = Float64,
                            )

#   ref_domain = unit_sq(;D=1)
#   domain = readjust_to_ref(domain, ref_domain)

    z, w = quad = quadrature(n)

    D = lagrange_poly_deriv_mat(z)

    z = T.(z) |> Field
    w = T.(w) |> Field

    grid = (z,)
    mass = DiagonalOp(w)
    grad = D |> MatrixOp

    space = SpectralSpace(domain, quad, grid, mass, grad, gather_scatter)
    return domain.mapping === nothing ? space : deform(space)
end

GaussLobattoLegendre1D(args...; kwargs...) = LagrangePolySpace1D(args...; quadrature=gausslobatto, kwargs...)
GaussLegendre1D(args...; kwargs...) = LagrangePolySpace1D(args...; quadrature=gausslegendre, kwargs...)
GaussChebychev1D(args...; kwargs...) = LagrangePolySpace1D(args...; quadrature=gausschebyshev, kwargs...)

function LagrangePolySpace2D(domain::AbstractDomain{<:Number,2} nr, ns;
                             quadrature = gausslobatto,
                             T = Float64,
                            )
    @assert domain isa BoxDomain "spectral polynomials need logically rectangular domains"

#   ref_domain = unit_sq(;D=2)
#   domain = readjust_to_ref(domain, ref_domain)

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

    global_numbering =

    space = SpectralSpace(grid, mass, grad, gather_scatter)
    space = domain.mapping === nothing ? space : deform(space)
end

GaussLobattoLegendre2D(args...; kwargs...) = LagrangePolySpace2D(args...; quadrature=gausslobatto, kwargs...)
GaussLegendre2D(args...; kwargs...) = LagrangePolySpace2D(args...; quadrature=gausslegendre, kwargs...)
GaussChebychev2D(args...; kwargs...) = LagrangePolySpace2D(args...; quadrature=gausschebyshev, kwargs...)
#
