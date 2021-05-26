#
import FastGaussQuadrature.gausslobatto
using SEM
using LinearAlgebra
using Parameters

#----------------------------------------------------------------------
# WORKS
Base.@kwdef mutable struct params
    n::Int
    fun::Function
    n1 = fun(n)
end
# WORKS
fun(n) = n+1
Base.@kwdef mutable struct params
    n::Int
    n1 = fun(n)
end
# DOESN'T WORK: fun not defined
Base.@kwdef mutable struct params
    n::Int
    fun::Function
    (n1,n2) = fun(n)
end

# DOESN'T WORK
using Parameters
@with_kw mutable struct params
    n::Int
    fun::Function
    (n1,n2) = fun(n)
end

# WORKS
using Parameters
@with_kw mutable struct params
    n::Int
    fun::Function
    n1 = fun(n)
end
#----------------------------------------------------------------------
Base.@kwdef mutable struct mesh
#@with_kw mutable struct mesh
    nr::Int
    ns::Int
    Ex::Int
    Ey::Int

    ifperiodic::Array{Char,1}
    usrgeom::Function

#   function mesh(nr,ns,Ex,Ey,usrgeom,ifperiodic)
#
    zr,wr = gausslobatto(nr)
    zs,ws = gausslobatto(ns)

#   Dr  = derivMat(zr)
#   Ds  = derivMat(zs)

#   # mappings
#   # Q: global -> local op, Q': local -> global
#   Qx = semq(Ex,nx,ifperiodic[1])
#   Qy = semq(Ex,nx,ifperiodic[2])
#   QQtx = Qx*Qx'
#   QQty = Qy*Qy'
#   mult = ones(nx*Ex,ny*Ey)
#   mult = gatherScatter(mult,QQtx,QQty)
#   mult = @. 1 / mult

#   xe,_ = semmesh(Ex,nx)
#   ye,_ = semmesh(Ey,ny)
#   x,y  = ndgrid(xe,ye)

#   # mesh deformation
#   x,y = usrgeom(x,y)

#   # jacobian
#   Jac,Jaci,rx,ry,sx,sy = jac(x,y,Dr,Ds)

#   # diagonal mass matrix
#   wx = kron(ones(Ex,1),wr)
#   wy = kron(ones(Ey,1),ws)

#   B  = Jac .* (wx*wy')
#   Bi = 1 ./ B

#   # Lapl solve
#   G11 = @. B * (rx * rx + ry * ry)
#   G12 = @. B * (rx * sx + ry * sy)
#   G22 = @. B * (sx * sx + sy * sy)

end
#----------------------------------------------------------------------
nothing
