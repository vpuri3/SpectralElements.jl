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
            # plt = meshplt(u,mshRef[])
            # display(plt)
            # println(mean(dfn.ν))
        end
        end
        return
    end

    return callback! ,setIC! ,setBC! ,setForcing! ,setVisc!
end

#----------------------------------#
Ex = 8; nr1 = 2;
Ey = 8; ns1 = 2;

ifperiodic = [false,false]
m1 = Mesh(nr1,ns1,Ex,Ey,ifperiodic)
bc = ['D','D','D','D']
diffuseU = Diffusion(bc,m1,Tf=0.0,dt=0.00)

simulate!(diffuseU,caseSetup!(diffuseU)...)
#----------------------------------#

utrue = diffuseU.fld.u

callback! ,setIC! ,setBC! ,setForcing! ,setVisc! = caseSetup!(diffuseU)

nu = [2.]
oper = Conv((3,3),1=>1,pad=1,stride=1)
# oper.weight[:,:,1,1].=-[0 1.1 0;1 -3.9 1;0 1 0]#./(.2)^2
p,re = Flux.destructure(oper)
ps = Params([p])

function model()
    dU = Zygote.ignore() do
        Diffusion(bc,m1,Tf=0.0,dt=0.00)
    end
    varVisc(ν,x,y,t) = @. nu+0*x

    function opLHS(u::Array,ν,bdfB,mshRef,M,p)
        # ν,bdfB,mshRef,M = args
        
        # lhs = p.*hlmz(u,ν,bdfB[1],mshRef[])
        
        lhs = ABu(mshRef[].Qy',mshRef[].Qx',mshRef[].mult.*u) # gather
        lhs = re(p)(reshape(lhs,size(lhs)...,1,1))[:,:,1,1]
        lhs = ABu(mshRef[].Qy,mshRef[].Qx,lhs) # scatter

        return lhs
    end
    LHSargs(dfn::Diffusion) = dfn.ν, dfn.tstep.bdfB, dfn.mshRef, dfn.fld.M, p
    dU.opLHS, dU.LHSargs = opLHS, LHSargs

    simulate!(dU,callback! ,setIC! ,setBC! ,setForcing! , setVisc!)
    upred = dU.fld.u
end
function loss()
    upred = model()
    mean(abs2,upred.-utrue)
end

opt = ADAM(5e-4)
Flux.train!(loss,ps,Iterators.repeated((), 300),opt, cb = () -> println(loss()))
grads = gradient(loss,ps)
grads[[p for p in ps]...]

plt = meshplt(model(),m1); display(plt)