#
###
# AbstractDomain interface #TODO
###

"""
args:
    AbstractDomain{T,D}
    direction < D
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
struct Interval{T<:Number,Ttag} <: AbstractDomain{T,1}
    x0::T
    x1::T
    tags::Ttag
    isperiodic::Bool

    function Interval(x0::Number, x1::Number,
                      tags = (nothing, nothing), isperiodic = false;
                      T = Float64)
        T = promote_type(T, eltype(x0), eltype(x1))
        new{T,Ttag}(T(x0), T(x1), tags, isperiodic)
end

function Interval(vec=[-1, 1], isperiodic, tags; T=T)
    Interval(vec..., isperiodic, tags; T=T)
end

isperiodic(dom::Interval, dir=1) = dom.isperiodic
end_points(dom::Interval, dir=1) = (dom.x0, dom.x1)

###
# Box Domain
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

function BoxDomain(x0s, x1s,
                   periodic = (false for i in 1:length(x0s)),
                   tags = (nothing for i in 1:length(x0s)),
                  )
    @assert length(x0s) == length(x1s)

    intervals = Interval.(vecs, periodic)
    BoxDomain(intervals)
end

struct Deformation{Tm}
    map::Tm
    isseparable::Bool
end

isseparable(def::Deformation) = def.isseparable

"""
Deform D-dimensional domain via mapping

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
