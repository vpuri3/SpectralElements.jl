#
using Flux, NNlib, DiffEqFlux, Optim, LinearAlgebra, Plots
#--------------------------------------#
np  = 2001 # data points
ng  = 31   # FEM  points
#--------------------------------------#
function udata(x)
    u   = @. sin(10*pi*x)*(-1<=x<=0)
    u  += @. sin(01*pi*x)*( 0< x<=1)
    u .+= 2.0
    return u
end

xp = Array(range(-1,stop=1,length=np))
xg = Array(range(-1,stop=1,length=ng)) # initialize equidistant FEM grid

up = udata(xp)
ug = udata(xg)
#--------------------------------------#
"""
    Initialize 3-layer shallow ReLU neural network.

    Layer 1 - fix FEM node location (activation layer)\n
    Layer 2 - linear op. to create FEM hat functions\n
    Layer 3 - scale hat functions and sum (linear)
"""
function initializeNN(x,u)
    n = length(x)
    NN = Chain(Dense(1,n,relu    )
              ,Dense(n,n,identity)
              ,Dense(n,1,identity)
              )

    NN[1].b .= -x # only trainable params
    NN[1].W .= 1.0
    # Nodal basis functions
    NN[2].b   .= 0.0
    NN[2].b[1] = 1.0
    # Coefficients
    NN[3].b .= 0.0

    updateNN!(NN) # NN[2].W .= Tridiagonal(hi,hh,hi); NN[3].W .= u'

    return NN
end
#--------------------------------------#
"""
    Given grid NN[1].b = -x, fix NN[2].W, NN[3].W to get
    adjusted lagrange basis functions, and scaling coeffs
"""
function updateNN!(NN)
    x = -NN[1].b
    u = udata(x) # replace with some linear solve

    h  = diff(x)
    hi = 1.0 ./ h 
    hh = zeros(size(x))
    hh[1:end-1] .-= hi
    hh[2:end  ] .-= hi

    NN[2].W .= Tridiagonal(hi,hh,hi)
    NN[3].W .= u'
    return
end
#--------------------------------------#
function makeplot()
    fp = model(xp')'

    plt = plot()
    plt = plot!(title="ReLU FEM",xlabel="x",ylabel="y")

    plt = plot!(xp,up,width=3,label="data")
    plt = plot!(xp,fp,width=3,label="prediction")
    plt = scatter!(xg,model(xg')',width=4,color="red",label=false)
    
    # FEM Basis
    for i=1:ng
        scale = model[3].W[i]
        scale = 1.0
        global plt = plot!(xp,scale .* model[2](model[1](xp'))[i,:]
                          ,width=2,color="black",label=false)
    end
    
    #savefig(plt,"reluFEM.png")
    display(plt)
    return plt
end
#--------------------------------------#
function loss()
    fp = model(xp')'
    loss = sum(abs2,fp .- up)
    return loss
end
#--------------------------------------#
function callback(;doplot=false)

    # apply gradient descent on model[1].b
    updateNN!(model)

    l2 = loss()
    fp = model(xp')'
    er = maximum(abs.(up.-fp))
    println("∞ norm: $er, loss: $l2")

    if(doplot) makeplot() end
end
#--------------------------------------#
model = initializeNN(xg,ug)
callback(doplot=true)
#res = Flux.train!(loss,Flux.params(),data,ADAM(0.05),cb=cb)
#--------------------------------------#
# create a restriction layer that enforces Dirichlet BC
# Neumann BC? Enforce via loss function

nothing
