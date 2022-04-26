#
###
# Lagrange polynomial function spaces
###

""" Lagrange polynomial spectral space """
struct LagrangePolynomialSpace{T,D,
                               Tdom<:AbstractDomain{T,D},
                               Tquad,Tgrid,
                               Tmass<:AbstractField{T,D},
                               Tderiv,
                              } <: AbstractSpectralSpace{T,D}
    domain::Tdom
    quadratures::Tquad
    grid::Tgrid
    mass_mat::Tmass
    deriv_mats::Tderiv
end

function LagrangePolynomialSpace1D(domain::AbstractDomain{<:Number,1}, n;
                                    quadrature = gausslobatto,
                                    T = Float64,
                                   )

    """ reset deformation to map from [-1,1]^D """ # TODO
#   ref_domain = unit_sq(;D=1)
#   domain = readjust_to_ref(domain, ref_domain)

    z, w = quadrature(n)

    D = lagrange_poly_deriv_mat(z)

    z = T.(z)
    w = T.(w)

    quadratures = ((z, w),)
    grid = (Field(z),)
    mass_mat = Field(w)
    deriv_mats = (D,)

    space = LagrangePolynomialSpace1D(
                                      domain, quadratures, grid,
                                      mass_mat, deriv_mats, 
                                     )
    domain.mapping === nothing ? space : deform(space)
end

GaussLobattoLegendre1D(args...; kwargs...) =
    LagrangePolySpace1D(args...; quadrature=gausslobatto, kwargs...)
GaussLegendre1D(args...; kwargs...) =
    LagrangePolySpace1D(args...; quadrature=gausslegendre, kwargs...)
GaussChebychev1D(args...; kwargs...) =
    LagrangePolySpace1D(args...; quadrature=gausschebyshev, kwargs...)

function LagrangePolySpace2D(domain::AbstractDomain{<:Number,2} nr, ns;
                             quadrature = gausslobatto,
                             T = Float64,
                            )
    msg = "spectral polynomials work with logically rectangular domains"
    @assert domain isa BoxDomain msg

    """ reset deformation to map from [-1,1]^D """ # TODO
#   ref_domain = unit_sq(;D=1)
#   domain = readjust_to_ref(domain, ref_domain)

    zr, wr = quadrature(nr)
    zs, ws = quadrature(ns)

    zr, wr = T.(zr), T.(wr)
    zs, ws = T.(zs), T.(ws)

    r, s = ndgrid(zr,zs)

    Dr = lagrange_poly_deriv_mat(zr)
    Ds = lagrange_poly_deriv_mat(zs)

    quadratures = ((z, w),)
    grid = (r, s)
    mass_mat = Field(w * w')
    deriv_mats = (Dr, Ds)

    space = LagrangePolynomialSpace1D(
                                      domain, quadratures, grid,
                                      mass_mat, deriv_mats, 
                                     )
    domain.mapping === nothing ? space : deform(space)
end

GaussLobattoLegendre2D(args...; kwargs...) =
    LagrangePolySpace2D(args...; quadrature=gausslobatto, kwargs...)
GaussLegendre2D(args...; kwargs...) =
    LagrangePolySpace2D(args...; quadrature=gausslegendre, kwargs...)
GaussChebychev2D(args...; kwargs...) =
    LagrangePolySpace2D(args...; quadrature=gausschebyshev, kwargs...)

grid(space::LagrangePolynomialSpace) = space.grid

function massOp(space::LagrangePolynomialSpace)
    @unpack mass_mat = space
    DiagonalOp(B)
end

function gradOp(space::LagrangePolynomialSpace{T,1})
    (Dr,) = space.deriv_mats

    Dx = MatrixOp(Dr)

    [Dx]
end

function gradOp(space::LagrangePolynomialSpace{T,2})
    (Dr, Ds) = space.deriv_mats

    nr = (space.quadratures[1])[1] |> length
    ns = (space.quadratures[2])[1] |> length

    Ir = Matrix(I, nr, nr) |> sparse
    Is = Matrix(I, ns, ns) |> sparse

    Dx = TensorProductOp2D(Dr, Is)
    Dy = TensorProductOp2D(Ir, Ds)

    [Dx
     Dy]
end

function gradOp(space::LagrangePolynomialSpace{T,3})
    (Dr, Ds, Dt) = space.deriv_mats

    nr = (space.quadratures[1])[1] |> length
    ns = (space.quadratures[2])[1] |> length
    nt = (space.quadratures[3])[1] |> length

    Ir = Matrix(I, nr, nr) |> sparse
    Is = Matrix(I, ns, ns) |> sparse
    It = Matrix(I, nt, nt) |> sparse

    Dx = TensorProductOp3D(Dr, Is, It)
    Dy = TensorProductOp3D(Ir, Ds, It)
    Dz = TensorProductOp3D(Ir, Is, Dt)

    [Dr
     Ds
     Dt]
end
#
