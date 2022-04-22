#
Base.@kwdef struct Interval{T,B,Te} <: AbstractDomain{T,1}
    end_points::Te = [-1e0,1e0]
    ifperiodic::Bool = false
end

function BoxDomain(vecs...;periodic=(false for i in 1:length(vecs)))
    intervals = Interval.(vecs, periodic)
end

struct BoxDomain{T,D,Ti} <: AbstractDomain{T,D}
    intervals::Ti

    function BoxDomain(intervals...)
        T = promote_type(eltype(intervals...))
        D = length(intervals)
        new{T,D,typeof(intervals)}(intervals)
    end
end

""" Interpolation operator between spaces """
struct Interp2D{T,Td1,Td2} <: AbstractOperator{T,2}
    domain1::Td1
    domain2::Td2
    function Interp2D() where{Tx,Ty}
        T = promote_type(Tx,Ty)
        new{T,typeof(x)}(Tx,Ty)
    end
end
#
