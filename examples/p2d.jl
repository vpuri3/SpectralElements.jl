#!/usr/bin/env julia

using Revise
using SEM
using LinearAlgebra,Plots, UnPack
#----------------------------------#
function caseSetup!(dfn::Diffusion)

    function setIC!(u,x,y,t)
        u .= 0.0
        return
    end

    function setBC!(ub,x,y,t)
        ub .= @. 0+0*x
        return
    end

    function setForcing!(f,x,y,t)
        f .= @. 1+0*x
        return
    end

    function setVisc!(ν,x,y,t)
        ν .= @. 1+0*x
        return
    end

    function callback!(dfn::Diffusion)
        @unpack fld,mshRef = dfn
        @unpack time, istep = dfn.tstep

        u = fld.u
        println("iter= $(istep[1]), time=$(time[1])")
        plt = meshplt(u,mshRef[])
        display(plt)
        return
    end

    return callback! ,setIC! ,setBC! ,setForcing! ,setVisc!
end

#----------------------------------#
Ex = 10; nr1 = 8;
Ey = 10; ns1 = 8;

ifperiodic = [false,true]
m1 = Mesh(nr1,ns1,Ex,Ey,ifperiodic,SEM.annulus)
bc = ['D','D','N','N']
diffuseU = Diffusion(bc,m1,Tf=0.0,dt=0.00)

simulate!(diffuseU,caseSetup!(diffuseU)...)
#----------------------------------#
