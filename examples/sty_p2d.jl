#!/usr/bin/env julia

using SEM, OrdinaryDiffEq, Flux
using LinearAlgebra,Plots
#----------------------------------#
Ex = 2; nr1 = 8;
Ey = 2; ns1 = 8;

m1 = Mesh(nr1,ns1,Ex,Ey,[true,true])
bc = ['D','D','D','D']
M  = generateMask(bc,m1)
opM(u) = m1.Bi .* u
opB(u) = gatherScatter(mass(u,m1),m1)
#----------------------------------#
function setIC!(u,x,y,t)
    u = 0.0
    return
end

function setBC!(ub,x,y,t)
    ub = 0+0*x
    return
end

function setForcing!(f,x,y,t)
    f = 1+0*x
    return
end

setVisc!(ν,x,y,t) = 1+0*x

#----------------------------------#
it = 0
function cond(u,t,integrator)
    global it
    cond = (it % 100) == 0
    it += 1
    return cond
end

function affect!(integrator)
    global it
    u = integrator.u
    p = integrator.p
    t = integrator.t
    dt = integrator.dt

    ut = utrue(m1.x,m1.y,t)
    er = norm(ut-u,Inf)
    println("Step=$it, Time=$t, dt=$dt, er=$er")
    plt = meshplt(u,m1)
    plt = plot!(zlims=(-1,1))
    display(plt)
    return
end

cb = DiscreteCallback(cond,affect!,save_positions=(false,false))

#----------------------------------#
function dudt(u,p,t)
    f = setForcing.(m1.x,m1.y,t)

    rhs = -lapl(u,m1) + mass(f,m1)
    rhs = gatherScatter(rhs,m1)

    dudt = pcg(rhs,opB;opM=opM,mult=m1.mult,ifv=false)

    return dudt
end

u0   = setIC(m1.x,m1.y)
dt   = 0.01
tspn = (0.0,1.0)

prob = ODEProblem(dudt,u0,tspn)
sol  = solve(prob,RK4();saveat=0.1,callback=cb)

err = sol.u - [utrue.(m1.x,m1.y,sol.t[i]) for i=axes(sol.t,1)]
ee = maximum.(abs.(err[i]) for i=axes(sol.t,1))
#----------------------------------#
return ee
