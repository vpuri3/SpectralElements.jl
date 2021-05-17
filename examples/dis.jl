#
using SEM

using FastGaussQuadrature
using DiffEqFlux,Flux
using Plots
#--------------------------------------#
function uapprox(x,p)
    cen,rad = p
    ua = zeros(size(x))
    for i=1:ne
        gllInterp!(ua,cen[i],rad[i],x)
    end
    return ua
end

function gllInterp!(uapprox,c,r,x)
    inElem(x) = @. (c-r) <= x <= (c+r)
    id = findall(inElem,x)
    xd = x[id]
    z = @. (c-r) + (1+zr)*0.5*2*r
    u = utrue(z)
    J = interpMat(xd,z)
    uapprox[id] += J*u
    return
end

function loss(p)
    ua = uapprox(xdata,p)
    l = sum(abs2,ua-udata)/length(xdata)
    return l,ua
end

#--------------------------------------#
function utrue(x)
    u = @. 2*x*(0<=x<=0.5) + 1*(0.5<x<=1.0)
    return u
end

xdata = linspace(0,1,100)
udata = utrue(xdata)

ne = 2
nx = 4

zr,zw = gausslobatto(nx)

centers = rand(ne)
radii   = rand(ne)

centers = [0.25,0.75]
radii   = [0.25,0.25]

p = (centers,radii)

#--------------------------------------#
function callback(p, l, pred)
  display(l)
  #plt = plot(pred, ylim = (0, 6))
  #display(plt)
  return false
end

#data = Iterators.repeated((),length(xdata))
#Flux.train!(loss,Flux.params(p),data,ADAM(0.05),cb=callback)

p1 = DiffEqFlux.sciml_train(loss, p,
                            ADAM(0.1),
                            cb = callback,
                            maxiters = 100)
#--------------------------------------#
loss(p)[1]
