#
""" 1D interval """
struct Interval{T,Te<:Vector{T}} <: AbstractDomain{T,1}
    end_points::Te
    periodic::Bool

    function Interval(vec=[-1, 1], periodic = false; T=Float64)
        vec = T.(vec)
        new{T, typeof(vec)}(vec, periodic)
    end
end

""" D-dimensional logically reectangular domain """
struct BoxDomain{T,D,Ti,Td} <: AbstractDomain{T,D}
    intervals::Ti

    function BoxDomain(intervals...)
        T = promote_type(eltype.(intervals)...)
        D = length(intervals)
        new{T,D,typeof(intervals)}(intervals)
    end
end

function BoxDomain(vecs...; periodic=(false for i in 1:length(vecs)))
    intervals = Interval.(vecs, periodic)
    BoxDomain(intervals)
end

struct DeformedDomain{T,D} <: AbstractDomain{T,D}
    ref_domain:Td
    mapping::Tm
end

""" Interpolation operator between spaces """
struct Interp2D{T,Td1,Td2} <: AbstractOperator{T,2}
    space1::Ts1 # or domain1/2?
    space2::Ts2
    function Interp2D() where{Tx,Ty}
        T = promote_type(Tx,Ty)
        new{T,typeof(x)}(Tx,Ty)
    end
end
#
