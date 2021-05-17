#!/usr/bin/env julia

# 1D possion solver

using SEM
using FastGaussQuadrature
using Plots, LinearAlgebra

n = 21

x,w = gausslobatto(n)
xo  = linspace(-1,1,20*n)

x  = 0.5.*(x .+1)
xo = 0.5.*(xo.+1)
w  = 0.5.*w

J = interpMat(xo,x)
D = derivMat(x)
B = Matrix(Diagonal(w))
A = D'*B *D

x,w,A,B,_,_ = linearFEM(n-1)

## CASE SETUP: solve: -\del^2 u = f + hom. dir BC
k  = @. 1.
ut = @. sin(k*pi*x)
f  = @. ut * (k*pi)^2
#f = @. 0*f + 1.

## restriction
R  = Matrix(I,n,n)
R  = R[2:end-1,:]

## full rank system
AA = R*A*R'
BB = R*B*R'
bb = R*B*f
uu = pcg(bb,AA)
xx = R*x
u  = R'*uu

## rank deficinet system
b  = R'*bb;
Al = R'*AA*R;
u  = pcg(b,Al);
#u  = pcg(f,B\Al);
#u  = pcg(b,Al,B);

println("er: ",norm(u - ut,Inf))
p=plot(x,u)
display(p)

#----------------------------------------------------------------------#
nothing
