#
# create Lienar FEM Hat functions with ReLU
#
# centred at x=0, width = 1.0 => spacing = 0.5
#
using Flux, NNlib, DiffEqFlux
using Plots
import SEM
#--------------------------------------#
x = SEM.linspace(-2,2,1000)


fm1 = @. 2.0relu(x+0.5)
fm0 = @. 4.0relu(x-0.0)
fp1 = @. 2.0relu(x-0.5)

hat = fm1 - fm0 + fp1

p = plot()
p = plot!(title="FEM Hat Function",xlims=[-1,1],ylims=[0,1.5])
p = plot!(x,hat,width=4,color=:black,style=:solid,legend=false)
savefig(p,"reluHat.png")

# ReLU
p=plot(title="ReLU"); p=plot!(x,relu.(x),width=4,color="black",legend=false,style=:solid); savefig(p,"relu.png")
