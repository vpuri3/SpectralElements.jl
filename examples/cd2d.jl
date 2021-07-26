#!/usr/bin/env julia

using SEM
using LinearAlgebra,Plots, UnPack
#----------------------------------#
kx=1.0
ky=1.0
ux=0.25
uy=0.0

function utrue(x,y,t)
    xx = @. x - ux*t
    yy = @. y - uy*t
    ut = @. sin(kx*pi*xx)*sin.(ky*pi*yy)
    return ut
end

function setIC!(u,x,y,t)
    u .= utrue(x,y,t)
    return
end

function setBC!(ub,x,y,t)
    ub .= 0.0
    return
end

function setForcing!(f,x,y,t)
    f .= 0.0
    return
end

function setVisc!(ν,x,y,t)
    ν .= 0.0
    return
end

function callback!(cdn::ConvectionDiffusion)
    @unpack fld,mshVRef = cdn
    @unpack time, istep = cdn.tstep

    ut = utrue(mshVRef[].x,mshVRef[].y,time[1])
    u  = fld.u
    er = norm(ut-u,Inf)
    println("Step $(istep[1]), Time=$(time[1]), er=$er")
    plt = meshplt(u,mshVRef[])
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
dt = 0.01
Temperature = ConvectionDiffusion(bc,m1,md,Tf=1.0,dt=dt)

Temperature.vx .= ux
Temperature.vy .= uy
for i=1:length(Temperature.fld.uh)
#   Temperature.fld.uh[i] .= utrue(m1.x,m1.y,-i*dt)
#   Temperature.exH[i]    .= -advect(Temperature.fld.uh[i]
#                                   ,Temperature.vx
#                                   ,Temperature.vy
#                                   ,m1,md
#                                   ,Temperature.JrVD
#                                   ,Temperature.JsVD)
end

#anim = Animation()
simulate!(Temperature,callback!,setIC!,setBC!,setForcing!,setVisc!)

#gif(anim,"ConvectionDiffusion.gif",fps=30)
#----------------------------------#

# grad test
if(false)
# gradient ok

x = m1.x
y = m1.y
u = utrue(x,y,0.0)

ux_t = @. (kx*pi)*cos(kx*pi*x)*sin(ky*pi*y)
uy_t = @. (ky*pi)*sin(kx*pi*x)*cos(ky*pi*y)
ux,uy = grad(u,m1)

ux = gatherScatter(ux,m1) .* m1.mult
uy = gatherScatter(uy,m1) .* m1.mult
meshplt(ux .- ux_t,m1)
meshplt(uy .- uy_t,m1)


# what else can go wrong? why is advection propogating
# at a faster speed than the true solution?
#
end
#----------------------------------#
