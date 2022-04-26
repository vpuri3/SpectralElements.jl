#
###
# AbstractDomain interface #TODO
###

"""
args:
    AbstractDomain
    direction
ret:
    Bool
"""
function isperiodic end

"""
args:
    AbstractDomain
    direction
ret:
    AbstractVector
"""
function end_points end

###
# Interval
###

""" 1D interval """
struct Interval{T,Te<:Vector{T}} <: AbstractDomain{T,1}
    end_points::Te
    periodic::Bool

    function Interval(vec=[-1, 1], periodic = false; T=Float64)
        vec = T.(vec)
        new{T, typeof(vec)}(vec, periodic)
    end
end

function isperiodic(domain::Interval, dir=1)
    domain.periodic
end

function end_points(domain::Interval, dir=1)
    domain.end_points
end

###
# Interval
###

""" D-dimensional logically reectangular domain """
struct BoxDomain{T,D,Ti} <: AbstractDomain{T,D}
    intervals::Ti

    function BoxDomain(intervals::Interval...)
        T = promote_type(eltype.(intervals)...)
        D = length(intervals)
        new{T,D,typeof(intervals)}(intervals)
    end
end

function BoxDomain(vecs::AbstractVector...;
                   periodic=(false for i in 1:length(vecs))
                  )
    intervals = Interval.(vecs, periodic)
    BoxDomain(intervals)
end

"""
Deform D-dimensional domain via map

x1,...,xD = map(r1, ..., rD)
"""
struct DeformedDomain{T,D,Td<:AbstractDomain{T,D}, Tm} <: AbstractDomain{T,D}
    domain::Td
    mapping::Tm
end

function deform(domain, mapping = nothing)
    DeformedDomain(domain, mapping)
end

function unit_square(;dims=D,T=Float64)
    BoxDomain([-one(T), one(T)])
end

function map_from_unit_sq(domain, ref_domain;D=D)

    if domain ≈ ref_domain
        return domain
    end

    xe = domain.end_points

    mapping = domain.mapping ==! nothing ? domain.mapping : (r -> r)
    mapping = mapping ∘ map
end
#
