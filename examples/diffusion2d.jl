#!/usr/bin/env julia

using Revise
using SEM
using LinearAlgebra,Plots

using UnPack
#----------------------------------#
function caseSetup!(dfn::Diffusion)

    kx=2.0
    ky=2.0
    function utrue(x,y,t)
        ut = @. sin(kx*pi*x)*sin.(ky*pi*y)
        return ut
    end

    function callback(dfn::Diffusion)
        @unpack fld,mshRef,time = dfn
        ut = utrue(mshRef[].x,mshRef[].y,time[1])
        u  = fld.u
        er = norm(ut-u,Inf)
        print("er: ",er,"\n")
        p = meshplt(u,mshRef[])
        display(p)
        return
    end

    function setBC!(ub,x,y,t)
        ub .= @. 0*x
        return
    end

    function setForcing!(f,x,y,t)
        f .= @. 1.0 + 0*x
#       ut = utrue(x,y,t)
#       f .= @. ut*((kx^2+ky^2)*pi^2)
        return
    end

    function setVisc!(ν,x,y,t)
        ν .= @. 1+0*x
        return
    end

    return setBC!, setForcing!, setVisc!, callback
end

#----------------------------------#
Ex = 8; nr1 = 8;
Ey = 8; ns1 = 8;

function deform(x,y) # deform [-1,1]^2
    x,y = SEM.annulus(0.5,1.0,2pi,x,y)
#   x = @. 0.5*(x+1)
#   y = @. 0.5*(y+1)
    return x,y
end

ifperiodic = [false,true]

m1 = Mesh(nr1,ns1,Ex,Ey,deform,ifperiodic)
bc = ['D','D','N','N']
diffuseU = Diffusion(bc,m1)

setBC!, setForcing!, setVisc!, callback = caseSetup!(diffuseU)

evolve!(diffuseU,setBC!,setForcing!,setVisc!,callback)
#----------------------------------#
nothing
