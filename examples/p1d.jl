#!/usr/bin/env julia

using SEM
using Plots,LinearAlgebra

n = 8
x,w = SEM.FastGaussQuadrature.gausslobatto(n)
xo  = SEM.linspace(-1,1,10*n)

J = SEM.interpMat(xo,x)
D = SEM.derivMat(x)

# neumann operators
B = diagm(w)
A = D'*B *D
C(v) = B * diagm(v) * D

## CASE SETUP
function utrue(x,v,t)
    xx = x - v*t
    return cos(2pi*xx)
#   return exp(-5*x*x)
end
f = @. 0+0*x
vl= 1.0
v = @. vl+0*x
ν = 0e-0
u = utrue.(x,vl,0.0)
id= Matrix(I,n,n)

#R = id[2:end-1,:] # double dirichlet
R = id[1:end-0,:] # periodic
Q = semq(1,n,true)

CFL = 0.1
dx = minimum(diff(x))
dt = CFL * dx / vl
println("CFL=$CFL, dt=$dt")

## global system (full rank)
xx = R*x
AA = R*A*R' .* ν
BB = R*B
CC = R*C(v)*R'
bb = BB*f
AC = AA + CC

## local system (rank deficient)
b  = R'*bb
Al = R'*AA*R
Bl = R'*BB
Cl = R'*CC*R
Hl = Al + Bl

#println("er: ",norm(u - ut,Inf))
p=plot(xo,J*u,width=3,ylims=(-1.5,1.5))
display(p)

uh = [zero(u) for i in 1:3] # histories
gh = [zero(u) for i in 1:3] # explicit term

T = 1.0
t = zeros(4,1)
nstep = Int(floor(T/dt))

for istep=1:nstep
    global u,rhs,Hl,bdfA,bdfB

    updateHist!(t)
    updateHist!(u,uh)

    t[1] += dt
    bdfA,bdfB = bdfExtK(t)

    gh[1] = -Cl*uh[1] # convection

    rhs = Bl * f
    for i=1:3
        gh[i] = -Cl*uh[i]                 # convection
        rhs  += -bdfB[1+i] .* (Bl *uh[i]) # histories
        rhs  +=  bdfA[i]   .* gh[i]       # explicit term
    end

    Hl = Al + bdfB[1]*Bl # Helmholtz op

    Hl  = Q*Q'*Hl
    rhs = Q*Q'*rhs
    u = pcg(rhs,Hl)

    ut = utrue.(x,vl,t[1])
    er = norm(u .- ut,Inf)
    println("Pointwise error: $er")

    plt = plot(xo,J*u,width=3)
    plt = plot!(title="Step $istep, Time $(t[1])")
    plt = plot!(ylims=(-1.5,1.5))
    display(plt)
end
println("CFL=$CFL, dt=$dt")
#----------------------------------------------------------------------#

#prob = ODEProblem(dudt,u0,tspn)
#sol  = solve(prob,SSPRK33(),dt=0.01,saveat=0.1,cb=callback)
#----------------------------------------------------------------------#
return
