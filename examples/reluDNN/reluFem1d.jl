#
using Flux, NNlib, DiffEqFlux, LinearAlgebra
using Plots
import SEM
#--------------------------------------#
d   = 1
Lb2 = 1e0
np  = 201 # sample points
ng  = 11  # grid points
#--------------------------------------#
# fem grid setup

xg = SEM.linspace(-Lb2,Lb2,ng)
xp = SEM.linspace(-Lb2,Lb2,np)

hg = diff(xg)
hi = 1.0 ./ hg
hh = zeros(ng)
hh[1:end-1] .-= hi
hh[2:end  ] .-= hi
#--------------------------------------#

NN = Chain(Dense(d ,ng,relu    )
          ,Dense(ng,ng,identity)
          ,Dense(ng,1 ,identity)
          )

NN[1].b .= -xg
NN[1].W .= 1.0
# Nodal basis functions
NN[2].b .= 0.0
NN[2].b[1] = 1.0
NN[2].W .= Tridiagonal(hi,hh,hi)
#
NN[3].b .= 0.0
NN[3].W .= 1.0 # function values
NN[3].W .= @. xg'^2

fp = NN(flatten(xp))'
#--------------------------------------#
# partitions

xpart = - NN[1].b ./ NN[1].W
fpart = NN(xpart')'
#--------------------------------------#
plt = plot()

plt = plot!(xp,fp)
plt = scatter!(xpart,fpart,color="red")
plt = plot!(title="f=NN(x)",xlabel="x",ylabel="y=NN(x)")
#plt = plot!(xlims=(-Lb2,Lb2))

display(plt)
#--------------------------------------#
#
