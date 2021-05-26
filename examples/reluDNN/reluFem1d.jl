#
using Flux, NNlib, DiffEqFlux, LinearAlgebra
using Plots
import SEM
#--------------------------------------#
d   = 1
Lb2 = 1e0
np  = 201 # sample points
ng  = 21  # grid points
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
# Coefficients
y(x) = @. 1.0 + sin(pi*x) #x^2
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
# Coefficients
NN[3].b .= 0.0
NN[3].W .= @. y(xg')

# create a restriction layer that enforces Dirichlet BC
# Neumann BC? Enforce via loss function

fp = NN(flatten(xp))'
# error
maximum(abs.(y(xp).-fp)) |> display
#--------------------------------------#
# partitions

xpart = - NN[1].b ./ NN[1].W
fpart = NN(xpart')'
#--------------------------------------#
plt = plot()

plt = plot!(xp,fp,width=3,legend=false)
plt = scatter!(xpart,fpart,width=4,color="red",label=false)
plt = plot!(title="y= DNN(x)",xlabel="x",ylabel="y")
#plt = plot!(xlims=(-Lb2,Lb2))

#--------------------------------------#
# plt basis just for kicks
for i=1:ng
    global plt = plot!(xp,NN[2](NN[1](xp'))[i,:],width=2,color="black",label=false)
end
#--------------------------------------#
savefig(plt,"reluFEM.png")
display(plt)
#--------------------------------------#
nothing
