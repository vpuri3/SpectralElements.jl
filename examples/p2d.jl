#!/usr/bin/env julia

using Revise
using SEM
using LinearAlgebra,Plots, UnPack
using Zygote, Flux
using Statistics
#----------------------------------#
function caseSetup!(dfn::Diffusion)

    function setIC!(u,x,y,t)
        u = 0.0 .*u
        return u
    end

    function setBC!(ub,x,y,t)
        ub = @. 0+0*x
        return ub
    end

    function setForcing!(f,x,y,t)
        f = @. 1+0*x
        return f
    end

    function setVisc!(ν,x,y,t)
        ν = @. 1+0*x
        return ν
    end

    function callback!(dfn::Diffusion)
        Zygote.ignore() do
        @unpack fld,mshRef = dfn
        @unpack time, istep = dfn.tstep

        u = fld.u
        if istep[1] == 1
            plt = meshplt(u,mshRef[])
            display(plt)
            println(mean(dfn.ν))
        end
        end
        return
    end

    return callback! ,setIC! ,setBC! ,setForcing! ,setVisc!
end

#----------------------------------#
Ex = 10; nr1 = 2;
Ey = 10; ns1 = 2;

ifperiodic = [false,false]
m1 = Mesh(nr1,ns1,Ex,Ey,ifperiodic)
bc = ['D','D','D','D']
diffuseU = Diffusion(bc,m1,Tf=0.0,dt=0.00)

ps = 0
simulate!(ps,diffuseU,caseSetup!(diffuseU)...)
#----------------------------------#

utrue = diffuseU.fld.u

callback! ,setIC! ,setBC! ,setForcing! ,setVisc! = caseSetup!(diffuseU)

nu = [.8]
ps = Params([nu])

function model()
    dU = Zygote.ignore() do
        Diffusion(bc,m1,Tf=0.0,dt=0.00)
    end
    function varVisc(ν,x,y,t)
        ν = @. nu+0*x
        return ν
    end
    simulate!(ps,dU,callback! ,setIC! ,setBC! ,setForcing! ,varVisc)
    upred = dU.fld.u
end
function loss()
    upred = model()
    sum(abs2,upred.-utrue)
end

opt = ADAM(1e-2)
Flux.train!(loss,ps,Iterators.repeated((), 100),opt)
