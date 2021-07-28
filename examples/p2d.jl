#!/usr/bin/env julia

using Revise
using SEM
using LinearAlgebra, Plots, UnPack, Setfield
using Zygote, Flux
using Statistics
#----------------------------------#

Ex = 10; nr1 = 2;
Ey = 10; ns1 = 2;

ifperiodic = [false,false]
m1 = Mesh(nr1,ns1,Ex,Ey,ifperiodic)
bc = ['D','D','D','D']

setIC(u,x,y,t) = 0.0 .*u
setBC(ub,x,y,t) = @. 0+0*x
setForcing(f,x,y,t) = @. 1+0*x
setVisc(ν,x,y,t) = @. 1+0*x

sch = DiffusionScheme(setIC,setBC,setForcing,setVisc)
dfn = Diffusion(bc,m1,sch,Tf=0.0,dt=0.00)

sim!(dfn)
utrue = dfn.fld.u

#----------------------------------#
# Learn ν #

ν0 = [2.]
varVisc(ν,x,y,t) = @. ν0+0*x

sch_ν = @set sch.setVisc = varVisc
dfn_ν = Diffusion(bc,m1,sch_ν,Tf=0.0,dt=0.00)

#----------------------------------#
l2g(u,msh) = ABu(msh.Qy',msh.Qx',msh.mult.*u)
g2l(u,msh) = ABu(msh.Qy,msh.Qx,u)
#----------------------------------#
# Learn laplace #

oper = Conv((5,5),1=>1,pad=2,stride=1)
oper.weight[:,:,1,1].=-[0 0 0 0 0;0 0 1 0 0;0 1 -4 1 0;0 0 1 0 0;0 0 0 0 0]./2
p0_lap,re_lap = Flux.destructure(oper)

function lapLearn(dfn::Diffusion)
    
    function opL(u,p0_lap,ν,mshRef)
        lhs = u.*ν
        lhs = l2g(lhs,mshRef[])
        lhs = re_lap(p0_lap)(reshape(lhs,size(lhs)...,1,1))[:,:,1,1]
        lhs = g2l(lhs,mshRef[]).*mshRef[].mult
        return lhs
    end
    
    return opL, (p0_lap,dfn.ν,dfn.mshRef)
end

sch_lap = @set sch.opLHS = lapLearn
dfn_lap = Diffusion(bc,m1,sch_lap,Tf=0.0,dt=0.00)

#----------------------------------#
# Learn solution op

m = Chain(Conv((3,3),1=>16,pad=1,stride=1,swish),
          Conv((3,3),16=>16,pad=1,stride=1,swish),
          Conv((3,3),16=>1,pad=1,stride=1))
p0_so,re_so = Flux.destructure(m)

function opLearn!(dfn::Diffusion)
    @unpack rhs,mshRef,fld = dfn
    @unpack u,ub = fld


    rhs = l2g(rhs,mshRef[])
    u = re_so(p0_so)(reshape(rhs,size(rhs)...,1,1))[:,:,1,1]
    u = g2l(u,mshRef[])
    u = mask(u,fld.M)

    u = u + ub
    @pack! dfn.fld = u
    return
end

sch_so = @set sch.solve! = opLearn!
dfn_so = Diffusion(bc,m1,sch_so,Tf=0.0,dt=0.00)

#----------------------------------#
# Training
#----------------------------------#

# Model + Loss

dfn_model = dfn_so
ps = Params([p0_so])

function model(dfn)
    sim!(dfn)
    upred = dfn.fld.u
end

function loss()
    upred = model(dfn_model)
    mean(abs2,upred.-utrue)
end

# Training

function cb()
    println(loss())
end

grads = gradient(loss,ps)
g = grads[[p for p in ps]...]

opt = ADAM(1e-3)
Flux.train!(loss,ps,Iterators.repeated((), 200),opt, cb = cb)

plt = meshplt(model(dfn_model),m1); plt = meshplt!(m1.x,m1.y,utrue,c=:blue); display(plt)