#!/usr/bin/env julia

using Revise
using SEM
using LinearAlgebra,Plots, UnPack
using Zygote, Flux
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
Ex = 4; nr1 = 2;
Ey = 4; ns1 = 2;

ifperiodic = [false,false]
m1 = Mesh(nr1,ns1,Ex,Ey,ifperiodic)
bc = ['D','D','D','D']
diffuseU = Diffusion(bc,m1,Tf=0.0,dt=0.00)

simulate!(diffuseU,caseSetup!(diffuseU)...)
#----------------------------------#

utrue = copy(diffuseU.fld.u)

nu = [0.8]
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
        ν .= @. nu+0*x
        return
    end

    function callback!(dfn::Diffusion)
        Zygote.ignore() do
        @unpack fld,mshRef = dfn
        @unpack time, istep = dfn.tstep

        u = fld.u
        println("iter= $(istep[1]), time=$(time[1])")
        plt = meshplt(u,mshRef[])
        display(plt)
        end
        return
    end

    return callback! ,setIC! ,setBC! ,setForcing! ,setVisc!
end

diffuseU = Diffusion(bc,m1,Tf=0.0,dt=0.00)
function loss()
    simulate!(diffuseU,caseSetup!(diffuseU)...)
    upred = diffuseU.fld.u
    sum(abs2,utrue.-upred)
end

grads = gradient(loss,Params([nu]))
grads[nu]