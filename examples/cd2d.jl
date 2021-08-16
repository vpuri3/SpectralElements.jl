#!/usr/bin/env julia

using SEM
using LinearAlgebra,Plots, UnPack
#----------------------------------#
kx=1.0
ky=1.0
ux=1.00
uy=0.0

function utrue(x,y,t)
    xx = @. x - ux*t
    yy = @. y - uy*t
    ut = @. sin(kx*pi*xx)*sin.(ky*pi*yy)
    return ut
end

function set0!(u,x,y,t)
    u .= utrue(x,y,t)
    return
end

function set∂!(ub,x,y,t)
    ub .= 0.0
    return
end

function setF!(f,x,y,t)
    f .= 0.0
    return
end

function setν!(ν,x,y,t)
    ν .= 0.0
    return
end

function callback!(cdn::ConvectionDiffusion)
    @unpack fld,mshV = cdn
    @unpack time, istep = cdn.tstep

    ut = utrue(mshV.x,mshV.y,time[1])
    u  = fld.u
    er = norm(ut-u,Inf)
    println("$(cdn.name), Step $(istep[1]), Time=$(time[1]), er=$er")
    plt = meshplt(u,mshV)
    plt = plot!(zlims=(-1,1))
    display(plt)
    #frame(anim)
    return
end

#----------------------------------#
Ex = 5; nr1 = 8; nrd = 8;
Ey = 5; ns1 = 8; nsd = 8;

ifperiodic=[true,false]
m1 = Mesh(nr1,ns1,Ex,Ey,ifperiodic)
md = Mesh(nrd,nsd,Ex,Ey,ifperiodic)
bc = ['N','N','D','D']

fld = Field(bc,m1)
vx = @. 0*m1.x + ux
vy = @. 0*m1.x + uy
tstep = TimeStepper(0.,1.,5e-3)
ps = ConvectionDiffusion("ps",fld,vx,vy,tstep,md,set0!,set∂!,setF!,setν!)

#anim = Animation()
simulate!(ps,callback!)

#gif(anim,"ConvectionDiffusion.gif",fps=30)
#----------------------------------#
nothing
