#!/usr/bin/env julia

using SEM
using LinearAlgebra,Plots, UnPack
#----------------------------------#
function caseSetup!(cdn::ConvectionDiffusion)

    kx=1.0
    ky=1.0
    ux=0.2
    uy=0.0

    function utrue(x,y,t)
        xx = @. x-ux*t
        yy = @. y-uy*t
        ut = @. sin(kx*pi*xx)*sin.(ky*pi*yy)
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
        f .= 0.0
        return
    end

    function setVisc!(ν,x,y,t)
        ν .= 0.0
        return
    end

    cdn.vx .= ux
    cdn.vy .= uy

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

    return callback! ,setIC! ,setBC! ,setForcing! ,setVisc!
end

#----------------------------------#
Ex = 10; nr1 = 8; nrd = 12;
Ey = 10; ns1 = 8; nsd = 12;

ifperiodic=[true,true]
m1 = Mesh(nr1,ns1,Ex,Ey,ifperiodic)
md = Mesh(nrd,nsd,Ex,Ey,ifperiodic)
bc = ['N','N','N','N']
Temperature = ConvectionDiffusion(bc,m1,md,Tf=1.0,dt=0.01)

#anim = Animation()
simulate!(Temperature,caseSetup!(Temperature)...)

#gif(anim,"ConvectionDiffusion.gif",fps=30)
#----------------------------------#
