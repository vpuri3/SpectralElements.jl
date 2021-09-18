#!/usr/bin/env julia

using SEM
using LinearAlgebra,Plots, UnPack
#----------------------------------#
kx=2.0
ky=2.0
kt=2.0
ν =1.0

function utrue(x,y,t)
    ut = @. sin(kx*pi*x)*sin.(ky*pi*y)*cos(kt*pi*t)
    return ut
end

function setIC!(u,x,y,t)
    u .= utrue(x,y,t)
    return
end

function setBC!(ub,x,y,t)
    ub .= @. 0+0*x
    return
end

function setForcing!(f,x,y,t)
    ut = utrue(x,y,t)
    f  .= @. ut*((kx^2+ky^2)*pi^2*ν)
    f .+= @. - sin(kx*pi*x)*sin(ky*pi*y)*sin(kt*pi*t)*(kt*pi)
    return
end

function setVisc!(ν,x,y,t)
    ν .= @. 1+0*x
    return
end

function callback!(dfn::Diffusion)
    @unpack fld,msh = dfn
    @unpack time, istep = dfn.tstep

    ut = utrue(msh.x,msh.y,time[1])
    u  = fld.u
    er = norm(ut-u,Inf)
    println("Step $(istep[1]), Time=$(time[1]), er=$er")
    plt = meshplt(u,msh)
    plt = plot!(zlims=(-1,1))
    display(plt)
    #frame(anim)
    return
end

#----------------------------------#
Ex = 5; nr1 = 8;
Ey = 5; ns1 = 8;

m1 = Mesh(nr1,ns1,Ex,Ey)
bc = ['D','D','D','D']
T = Diffusion(bc,m1,Tf=1.0,dt=0.01) # temperature
#----------------------------------#
function dudt!(dudt,u,msh::Mesh)

    @unpack x,y = msh
    # can play directly with full solution (uh + ub)
    # since we're no longer solving a BVP (explicit time method)

    dudt = lapl(u,msh) + f(msh)
    return dudt
end

prob = ODEProblem(dudt!,)
sol  = solve(prob,SSPRK33(),dt=0.01,saveat=0.25)
#----------------------------------#
