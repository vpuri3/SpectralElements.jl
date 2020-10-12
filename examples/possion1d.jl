#!/usr/bin/env julia

# struct Name
#   field::OptionalType
#   ...
# end

using SEM
using FastGaussQuadrature
using Plots, LinearAlgebra

linspace(zi::Number,ze::Number,n::Integer) = range(zi,stop=ze,length=n)
cumprod(A::AbstractArray) = Base.cumprod(A, dims=1)
cumprod(A::AbstractArray, d::Int) = Base.cumprod(A, dims=d)
sum(A::AbstractArray, n::Int) = Base.sum(A, dims=n)
sum(A) = Base.sum(A)
flipdim(A, d) = reverse(A, dims=d)


n = 16

x ,w  = gausslobatto(n)
xo    = linspace(-1,1,20*n)

x  = 0.5.*(x .+1)
xo = 0.5.*(xo.+1)
w  = 0.5.*w

J = interpMat(xo,x)
D = derivMat(x)
B = Diagonal(w)

k = 1.
ut = sin.(k*pi*x)
f = (k*pi)^2 .* ut
A = D'*B *D
u = A[2:end-1,2:end-1] \ (B*f)[2:end-1]
u = [0;u;0]

println(norm(u - ut,Inf))

plot(xo,J*u)
plot!(xo,J*ut)
