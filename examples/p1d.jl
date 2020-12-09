#!/usr/bin/env julia

# single element 1D possion solver

using SEM
using FastGaussQuadrature
using Plots, LinearAlgebra

import Krylov

#linspace(zi::Number,ze::Number,n::Integer) = range(zi,stop=ze,length=n)
#cumprod(A::AbstractArray) = Base.cumprod(A, dims=1)
#cumprod(A::AbstractArray, d::Int) = Base.cumprod(A, dims=d)
#sum(A::AbstractArray, n::Int) = Base.sum(A, dims=n)
#sum(A) = Base.sum(A)
#flipdim(A, d) = reverse(A, dims=d)

n = 12

x,w = gausslobatto(n)
xo  = linspace(-1,1,20*n)

x  = 0.5.*(x .+1)
xo = 0.5.*(xo.+1)
w  = 0.5.*w

J = interpMat(xo,x);
D = derivMat(x);
B = Matrix(Diagonal(w));
A = D'*B *D
R = Matrix(I,n,n);
R = R[2:end-1,:];

## CASE SETUP: solve: -\del^2 u = f + hom. dir BC
k  = @. 1.
ut = @. sin(k*pi*x)
f  = @. ut * (k*pi)^2;
f = @. 0*f + 1.;

## full rank system
ff = R*f;
AA = R*A*R';
BB = R*B*R';
bb = R*B*f;
#uu = cg(ff,R*(B\A)*R');
uu = cg(ff,BB\AA); # what's with the difference in performance???
uu = cg(bb,AA); # == cg(BB*ff,AA);
u  = R'*uu;
xx = R*x;

## rank deficinet system
b  = R'*bb;
Al = R'*AA*R;
u  = cg(f,B\Al);
u  = cg(b,Al);

u  = cgSEM(b,[],[],[],Al);
u  = cgSEM(b,[],[],B ,Al);

#println("er: ",norm(u - ut,Inf))
#p=plot(xo,J*u)
#display(p);

#----------------------------------------------------------------------#
#ll,vv=eigen(AA,BB);
#ll,vv=eigen(AA);
#v = R' * vv;
#Jv = J*v;
#p=plot(xo,Jv[:,1]);
#for i=2:n-2
#    plot!(xo,Jv[:,i])
#end
#display(p);
e1 = v[:,1];

#display(ut ./ e1);
#----------------------------------------------------------------------#
nothing
