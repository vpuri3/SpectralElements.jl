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
kx=1.0
ky=1.0
kt=1.0
ν =1.0

utrue(x,y,t) = @. sin(kx*pi*x)*sin.(ky*pi*y)*cos(kt*pi*t)
setIC(x,y)   = utrue(x,y,0.0)

function setForcing(x,y,t)
    ut = utrue(x,y,t)
    f  = ut*((kx^2+ky^2)*pi^2*ν)
    f -= sin(kx*pi*x)*sin(ky*pi*y)*sin(kt*pi*t)*(kt*pi)
#   f  = zero(x)
end

#----------------------------------#
function cond(u,t,integrator)
    istep = integrator.iter
    cond = (istep % 1) == 0
    return cond
end

function affect!(integrator)
    istep = integrator.iter
    u = integrator.u
    p = integrator.p
    t = integrator.t
    dt = integrator.dt

    ut = utrue(m1.x,m1.y,t)
    er = norm(ut-u,Inf)
    println("Step=$istep, Time=$t, dt=$dt, er=$er")
    plt = meshplt(u,m1)
    plt = plot!(zlims=(-1,1))
    display(plt)
    return
end

cb = DiscreteCallback(cond,affect!,save_positions=(false,false))

#----------------------------------#
function dudt!(du,u,p,t)
    f = setForcing.(m1.x,m1.y,t)

    rhs = -lapl(u,m1) + mass(f,m1)
    rhs = gatherScatter(rhs,m1)

    pcg!(du,rhs,opB;opM=opM,mult=m1.mult,ifv=false)

    return du
end

u0   = setIC(m1.x,m1.y)
dt   = 0.01
tspn = (0.0,1.0)

func = ODEFunction(dudt!)
prob = ODEProblem(func,u0,tspn)
sol  = solve(prob, Rodas5(); saveat=0.1, callback=cb)

err = sol.u - [utrue.(m1.x,m1.y,sol.t[i]) for i=axes(sol.t,1)]
ee = maximum.(abs.(err[i]) for i=axes(sol.t,1))
#----------------------------------#
return ee
