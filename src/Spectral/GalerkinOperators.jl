#
###
# Gather-Scatter Operators - enforce continuity/ periodicity
###

struct GatherScatter{D} <: AbstractGatherScatterOperator{Bool,D}
    global_numbering
    operator # implementation
end

function GatherScatter(space::AbstractSpectralSpace{<:Number,D}) where{D}
    periodic = space.domain.periodic

    if !prod(periodic)
        return IdentityOp{D}()
    end

    (n,) = space.npoints
    Q = Matrix(I,n, n-1) |> sparse
    Q[end,1] = 1

    QQt = Q * Q'
    MatrixOp(QQt)

    GatherScatter{1}(local2gl)
end

###
# Boundary Condition application
###

struct DirichletBC end

struct BoundaryCondition{T,D} <: AbstractBoundaryCondition{T,D}
    tags
    type # dirichlet, neumann
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
 bc = (:Dirichlet,:Neumann,:Dirichlet,:Dirichlet) at (rmin, rmax, smin, smax)

 :Dirichlet = Dirichlet = zeros ∂Ω data\n
 :Neumann   = Neumann   = keeps ∂Ω data

 A periodic mesh overwrites 'D' to 'N' in direction of periodicity.

 To achieve inhomogeneous Dirichlet condition, apply the formulation
 u = ub + uh, where uh is homogeneous part, and ub is an arbitrary
 smooth function on Ω. Then, solve for uh
"""
function generate_mask(tags, space::AbstractSpace{<:Number,2})
    (nr, ns,) = space.npoints

    periodic = isperiodic(space.domain)

    Ix = sparse(I,nr,nr)
    Iy = sparse(I,ns,ns)

    ix = collect(1:(nr))
    iy = collect(1:(ns))

    if(bc[1] == :Dirichlet) ix = ix[2:end]   end
    if(bc[2] == :Dirichlet) ix = ix[1:end-1] end
    if(bc[3] == :Dirichlet) iy = iy[2:end]   end
    if(bc[4] == :Dirichlet) iy = iy[1:end-1] end

    if(periodic[1]) ix = collect(1:(nr)); end
    if(periodic[2]) iy = collect(1:(ns)); end

    Rx = Ix[ix,:]
    Ry = Iy[iy,:]

    M = diag(Rx'*Rx) * diag(Ry'*Ry)'
    M = Array(M) .== true

    return M
end

function applyBC!(u::AbstractField{<:Number,D}, bc::BoundaryCondition{<:Number,D}) where{D}

    return u
end

function applyBC!()
end
#