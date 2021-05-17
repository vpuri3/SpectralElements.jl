#
using Flux, NNlib, DiffEqFlux
using Plots
import SEM
#--------------------------------------#
d   = 2
Lb2 = 1e1
np  = 100
nl  = 10
#--------------------------------------#
xx = SEM.linspace(-Lb2,Lb2,np)

if(d==1)
    x = SEM.ndgrid(xx)
    xy = Array{Float64}(undef,np,d)
    xy .= x
elseif(d==2)
    x,y = SEM.ndgrid(xx,xx)
    xy = Array{Float64}(undef,np,np,d)
    xy[:,:,1] .= x
    xy[:,:,2] .= y
end

xy = flatten(xy)'

NN = Chain(Dense(d ,nl,relu    ,initb=Flux.glorot_normal)
#         ,Dense(nl,nl,relu    ,initb=Flux.glorot_normal)
          ,Dense(nl,1 ,identity,initb=Flux.glorot_normal)
          )

f = NN(xy)
f = reshape(f,size(x))
#--------------------------------------#
# partitions
if(d==1)
    # first layer discontinuitites
    xpart = - NN[1].b ./ NN[1].W
    fpart = NN(xpart')'
elseif(d==2)
    # first layer discontinuities
    #NN[1].W * [x,y] + NN[1].b == 0
    x1 = -Lb2 * ones(size(NN[1].b))
    x2 =  Lb2 * ones(size(NN[1].b))
    y1 = @. (-NN[1].b - x1*NN[1].W[:,1]) / NN[1].W[:,2]
    y2 = @. (-NN[1].b - x2*NN[1].W[:,1]) / NN[1].W[:,2]

    xpart = vcat(x1',x2')
    fpart = vcat(y1',y2')
end
#--------------------------------------#
plt = plot()
if(d==1)
    plt = plot!(x,f)
    plt = scatter!(xpart,fpart,color="red")
    plt = plot!(title="f=NN(x)",xlabel="x",ylabel="y=NN(x)")
    plt = plot!(xlims=(-Lb2,Lb2))
elseif(d==2)
#   plt = heatmap!(f)
    plt = plot!(xpart,fpart,legend=false,color="black")
    plt = plot!(title="f=NN(x,y)",xlabel="x",ylabel="y")
#   plt = plot!(xlims=(-Lb2,Lb2),ylims=(-Lb2,Lb2))
end

display(plt)
#--------------------------------------#
#
