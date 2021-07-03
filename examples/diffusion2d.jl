#!/usr/bin/env julia

using Revise
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
#       ub .= utrue(x,y,t)
        return
    end

    function setForcing!(f,x,y,t)
        ut = utrue(x,y,t)
        f  .= @. ut*((kx^2+ky^2)*pi^2*ν)
        f .+= @. - sin(kx*pi*x)*sin(ky*pi*y)*sin(kt*pi*t)*(kt*pi)
#       f  .= @. 1+0*x
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
        p = meshplt(u,mshRef[])
        display(p)
        return
    end

    return setIC!,setBC!,setForcing!,setVisc!,callback!
end

#----------------------------------#
Ex = 10; nr1 = 8;
Ey = 10; ns1 = 8;

function deform(x,y) # deform [-1,1]^2
#   x,y = SEM.annulus(0.5,1.0,2pi,x,y)
    x = @. 0.5*(x+1)
    y = @. 0.5*(y+1)
    return x,y
end

ifperiodic = [false,true] # overwrites bc from 'D' to 'N' in periodic dir
ifperiodic = [false,false]

m1 = Mesh(nr1,ns1,Ex,Ey,deform,ifperiodic)
bc = ['D','D','N','N']
bc = ['D','D','D','D']
diffuseU = Diffusion(bc,m1,Tf=0.0,dt=0.00)

simulate!(diffuseU,caseSetup!(diffuseU)...)
#----------------------------------#
nothing
