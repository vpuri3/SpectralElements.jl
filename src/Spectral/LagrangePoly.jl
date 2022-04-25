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

    z, w = quadrature(n)

    D = lagrange_poly_deriv_mat(z)

    z = T.(z) |> Field
    w = T.(w) |> Field

    grid = (z,)
    mass = DiagonalOp(w)
    grad = D |> MatrixOp

    space = SpectralSpace(grid, mass, grad, gather_scatter)
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

###
# Gather-Scatter Operators
###

struct GatherScatter{D} <: AbstractGatherScatterOperator{Bool,D}
    global_numbering
    operator
end

function GatherScatter1D(space::AbstractSpace{T,1})
    periodic = space.domain.periodic

    QQt = if periodic
        n = length(space.grid(x))
        Q = Matrix(I,n, n-1) |> sparse
        Q[end,1] = 1

        QQt = Q * Q'
        MatrixOp(QQt)
    else
        IdentityOp{1}()
    end

    GatherScatter{1}(local2gl)
end

###
# Interpolation operators
###

""" Interpolation operator between spaces """
struct Interp2D{T,Td1,Td2} <: AbstractOperator{T,2}
    space1::Ts1 # or domain1/2?
    space2::Ts2

    # need general purpose implementation
    # need guarantees that domains match

    function Interp2D(space1, space2) where{Tx,Ty}
        T = promote_type(eltype.(space1, space2)...)
        new{T,typeof(space1),typeof(space2)}(Tx,Ty)
    end
end

""" Point-to-point interpolant when domains match """
function LagrangeInterpolant1D(space1::AbstractSpace{T,1},
                               space2::AbstractSpace{T,1})
    @assert space1.domain ≈ space2.domain

    r1 = grid(space1)
    r2 = grid(space2)

    Jr = lagrange_poly_interp_mat(r2, r1)

    MatrixOp(Jr)
end

function LagrangeInterpolant2D(space1::AbstractSpace{T,2},
                               space2::AbstractSpace{T,2})
    @assert space1.domain ≈ space2.domain

    r1 = grid(space1)[1] # get 1D grid
    r2 = grid(space2)[1]

    s1 = grid(space1)[2]
    s2 = grid(space2)[2]

    Jr = lagrange_poly_interp_mat(space2.grid, space1.grid)
    Js = lagrange_poly_interp_mat(space2.grid, space1.grid)

    TensorProduct2DOp(Jr, Js)
end

###
# BC implementation
###

struct BoundaryCondition{T,D} <: AbstractBoundaryCondition{T,D}
    tags # dirichlet, neumann
    dirichlet_func! # (ub, space) -> mul!(ub, I, false)
    neumann_func!
    mask # implementation
end

function BoundaryCondition(tags, space::AbstractSpace<:Number,2;
                           dirichlet_func! =nothing, neumann_func! = nothing)

    mask = generate_mask(tags, space)

    BoundaryCondition()
end

"""
 bc = ['D','N','D','D'] === BC at [xmin,xmax,ymin,ymax]

 :Dirichlet : Hom. Dirichlet = zeros ∂Ω data\n
 :Neumann   : Hom. Neumann   = keeps ∂Ω data

 A periodic mesh overwrites 'D' to 'N' in direction of periodicity.

 To achieve inhomogeneous Dirichlet condition, apply the formulation
 u = ub + uh, where uh is homogeneous part, and ub is an arbitrary
 smooth function on Ω. Then, solve for uh
"""
function generate_mask(tags, space::AbstractSpace{<:Number,2})
    nr = length()
    ns = length()

    periodic = space.domain.periodic

    Ix = Matrix(I,nr,nr) |> sparse
    Iy = Matrix(I,ns,ns) |> sparse

    ix = collect(1:(Ex*nr))
    iy = collect(1:(Ey*ns))

    if(bc[1]==:Dirichlet) ix = ix[2:end]   end
    if(bc[2]==:Dirichlet) ix = ix[1:end-1] end
    if(bc[3]==:Dirichlet) iy = iy[2:end]   end
    if(bc[4]==:Dirichlet) iy = iy[1:end-1] end

    if(periodic[1]) ix = collect(1:(Ex*nr)); end
    if(periodic[2]) iy = collect(1:(Ey*ns)); end

    Rx = Ix[ix,:]
    Ry = Iy[iy,:]

    M = diag(Rx'*Rx) * diag(Ry'*Ry)'
    M = Array(M) .== true

    return mask
end

function applyBC!(u::AbstractField{<:Number,D}, bc::BoundaryCondition{<:Number,D}) where{D}

    return u
end
#
