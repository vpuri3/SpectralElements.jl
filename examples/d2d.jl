#!/usr/bin/env julia

using SEM
using LinearAlgebra,Plots, UnPack
#----------------------------------#
function caseSetup!(dfn::Diffusion)

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
        @unpack fld,mshRef = dfn
        @unpack time, istep = dfn.tstep

        ut = utrue(mshRef[].x,mshRef[].y,time[1])
        u  = fld.u
        er = norm(ut-u,Inf)
        println("iter= $(istep[1]), time=$(time[1]), er=$er")
        plt = meshplt(u,mshRef[])
        display(plt)
        return
    end

    return callback! ,setIC! ,setBC! ,setForcing! ,setVisc!
end

#----------------------------------#
Ex = 10; nr1 = 8;
Ey = 10; ns1 = 8;

m1 = Mesh(nr1,ns1,Ex,Ey)
bc = ['D','D','D','D']
Temperature = Diffusion(bc,m1,Tf=1.0,dt=0.01)

simulate!(Temperature,caseSetup!(Temperature)...)
#----------------------------------#
