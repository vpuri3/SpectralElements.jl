#
###
# Gather-Scatter Operators - enforce continuity/ periodicity
###

struct GatherScatter{D} <: AbstractGatherScatterOperator{Bool,D}
    global_numbering
    operator # implementation
end

function GatherScatter(space::AbstractSpace{<:Number,D}) where{D}
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

    r1 = space1.quad[1].z
    r2 = space2.quad[1].z

    s1 = space1.quad[2].z
    s2 = space1.quad[2].z

    Jr = lagrange_poly_interp_mat(space2.grid, space1.grid)
    Js = lagrange_poly_interp_mat(space2.grid, space1.grid)

    TensorProduct2DOp(Jr, Js)
end

###
# Boundary Condition application
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
