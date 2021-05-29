#!/usr/bin/env julia

using Revise
using SEM
using LinearAlgebra,Plots,UnPack
using BenchmarkTools

#----------------------------------#
Ex = 8; nr1 = 8;
Ey = 8; ns1 = 8;

function deform(x,y)
    x,y = SEM.annulus(0.5,1.0,2pi,x,y)
#   x = @. 0.5*(x+1)
#   y = @. 0.5*(y+1)
    return x,y
end

ifperiodic = [false, true]

m1 = Mesh(nr1,ns1,Ex,Ey,deform,ifperiodic)

#----------------------------------#
# case setup
#----------------------------------#

x1 = m1.x
y1 = m1.y

ν  = @. 1+0*x1
f  = @. 1+0*x1
ub = @. 0+0*x1

M1 = generateMask(['D','D','N','N'],m1)
Mb = generateMask(['N','N','N','N'],m1)

function F(u,msh::Mesh)
    @unpack x,y = msh
    Fu = @. u^2
    return Fu
end

function dF(u,msh::Mesh)
    @unpack x,y = msh
    dFdu = @. 2u
    return dFdu
end

#----------------------------------#
# ops for PCG
#----------------------------------#

function opLapl(v)
    Au = lapl(v,m1)
    Au = gatherScatter(Au,m1.QQtx,m1.QQty)
    Au = mask(Au,M1)
    return Au
end

function opFDM(v) # preconditioner
    return v
end
#----------------------------------#
b =     mass(f ,m1)
b = b - lapl(ub,m1)

b = mask(b ,M1)
b = gatherScatter(b,m1)

u = copy(b)
#u = pcg(b,opLapl,opM=opFDM,mult=m1.mult,ifv=true)
@time pcg!(u,b,opLapl,opM=opFDM,mult=m1.mult,ifv=true)

u = u + ub
#----------------------------------#
plt = meshplt(u,m1)
display(plt)
#----------------------------------#
nothing
