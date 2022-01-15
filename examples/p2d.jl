#!/usr/bin/env julia

using SpectralElements, Flux, LinearAlgebra, Plots, UnPack
import Zygote
#----------------------------------#
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
    @unpack fld,msh = dfn
    @unpack time, istep = dfn.tstep

    u = fld.u
    println("iter= $(istep[1]), time=$(time[1])")
    plt = meshplt(u,msh)
    display(plt)
    return
end

#----------------------------------#
Ex = 5; nr1 = 8;
Ey = 5; ns1 = 8;

ifperiodic = [false,true]
m1 = Mesh(nr1,ns1,Ex,Ey,ifperiodic,SpectralElements.annulus)
bc = ['D','D','N','N']
diffuseU = Diffusion(bc,m1,Tf=0.0,dt=0.00)

ps = 0

simulate!(diffuseU,callback!,setIC!,setBC!,setForcing!,setVisc!)

#----------------------------------#
#utrue = diffuseU.fld.u
#
#nu = [0.8]
#ps = Flux.params(nu)
#
#function model()
#    dU = Zygote.ignore() do
#        Diffusion(bc,m1,Tf=0.0,dt=0.0)
#    end
#    function varVisc!(ν,x,y,t)
#        ν .= @. nu + 0*x
#        return ν
#    end
#    simulate!(dU,callback!,setIC!,setBC!,setForcing!,varVisc!)
#    return dU.fld.u
#end
#function loss()
#    upred = model()
#    return Flux.Losses.mse(upred,utrue)
#end
#
#opt = Flux.Optimise.ADAM(1e-2)
#gs = Flux.gradient(()->loss(),ps)

#Flux.train!(loss,ps,Iterators.repeated((),100),opt)
#----------------------------------#
