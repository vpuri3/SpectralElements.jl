#!/usr/bin/env julia

using Revise,SEM
using FastGaussQuadrature
using Plots,LinearAlgebra

n = 16
x,w = gausslobatto(n)
xo  = linspace(-1,1,20*n)

J = interpMat(xo,x)
D = derivMat(x)

# neumann operators
B = diagm(w)
A = D'*B *D
C(v) = B * diagm(v) * D

## CASE SETUP
function utrue(x,v,t)
    xx = @. x - v*t
    return @. sin(2pi*xx)
end
f = @. 0+0*x
vl= 1.0
v = @. vl+0*x
ν = 0e-0
u = utrue(x,vl,0.0)
id= Matrix(I,n,n)

#R = id[2:end-1,:] # double dirichlet
R = id[1:end-0,:] # periodic
Q = semq(1,n,true)

CFL = 0.1
dx = minimum(diff(x))
dt = dx * CFL / vl
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

    Hl = Al + bdfB[1]*Bl

    Hl  = Q*Q'*Hl
    rhs = Q*Q'*rhs
    u = pcg(rhs,Hl)

    ut = utrue(x,vl,t[1])
    er = norm(u .- ut,Inf)
    println("Pointwise error: $er")

    plt = plot(xo,J*u,width=3)
    plt = plot!(title="Step $istep, Time $(t[1])")
    plt = plot!(ylims=(-1.5,1.5))
    display(plt)
end
#----------------------------------------------------------------------#
nothing
