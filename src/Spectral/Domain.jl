abstract type AbstractDomain{T,D} end

struct Interval{T,B,I} <: AbstractDomain{T,1}
    pts::Vector{T}
    bcs::B
    interior::I
end

function Interval(x)

struct BoxDomain2D{T,I1,I2,B} <: AbstractDomain{T,2}
    interval1::I1
    interval2::I2
    bcs::B
end

struct BoxDomain3D{T,I0,I1,I2} <: AbstractDomain{T,3}
    interval1::I0
    interval2::I1
    interval3::I1
    bcs::B
end

"""
 Quadilateral Domain
"""
struct QuadDomain2D{T,Tx} <: AbstractDomain{T,2}
  x::Tx
  y::Tx
  xperiodic::Bool
  yperiodic::Bool
  function QuadDomain2D(x::AbstractField{Tx,2},
                        y::AbstractField{Ty,2},
                        xperiodic::Bool = false,
                        yperiodic::Bool = false) where{Tx,Ty}
    T = promote_type(Tx,Ty)
    new{T,typeof(x)}(Tx,Ty,xperiodic,yperiodic)
  end
end

"""
 Interpolating Operator
"""
struct Interp2D{T,Tx} <: AbstractDomain{T,2}
  domain1
  domain2
  function Interp2D(x::AbstractField{Tx,2},
                    y::AbstractField{Ty,2}) where{Tx,Ty}
    T = promote_type(Tx,Ty)
    new{T,typeof(x)}(Tx,Ty)
  end
end

