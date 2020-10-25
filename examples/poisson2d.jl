#!/usr/bin/env julia

# struct Name
#   field::OptionalType
#   ...
# end

using .SEM

using FastGaussQuadrature, LinearOperators
using Plots, LinearAlgebra
using SmoothLivePlot
using DiffEqFlux, Flux, Zygote, Optim

using Krylov
linspace(zi::Number,ze::Number,n::Integer) = range(zi,stop=ze,length=n)

nx1 = 32; nxd = Int(ceil(1.5*nx1)); nxo = 10*nx1;
ny1 = 32; nyd = Int(ceil(1.5*ny1)); nyo = 10*ny1;

zr1,wr1 = gausslobatto(nx1); zrd,wrd = gausslobatto(nxd); zro=linspace(-1,1,nxo);
zs1,ws1 = gausslobatto(ny1); zsd,wsd = gausslobatto(nyd); zso=linspace(-1,1,nyo);

Jr1d = interpMat(zrd,zr1); Jr1o = interpMat(zro,zr1);
Js1d = interpMat(zsd,zs1); Js1o = interpMat(zso,zs1);

Dr1 = derivMat(zr1);
Ds1 = derivMat(zs1);

# nodal grid
r1,s1 = ndgrid(zr1,zs1);
rd,sd = ndgrid(zrd,zsd);

x1,y1 = r1,s1;          # solve on nodal grid for now.
xd,yd = rd,sd;

Jac1  = ones(nx1,ny1);  # unit jacobian
Jacd  = ones(nxd,nyd);

# diagonal mass matrices
B1  = (wr1 * ws1') .* Jac1;
Bd  = (wrd * wsd') .* Jacd;
Bi1 = 1 ./ B1;
Bid = 1 ./ Bd;

#----------------------------------------------------------------------#
# all hom. dirichlet BC
#----------------------------------------------------------------------#
m1=[0;ones(nx1-2);0]; m2=[0;ones(ny1-2);0];
M = m1 * m2';

#----------------------------------------------------------------------#
# case setup
#----------------------------------------------------------------------#

function setup(visc, f)
    # set up Laplace operator
    rx1 = @. 1+0*x1;
    ry1 = @. 0+0*x1;
    sx1 = @. 0+0*x1;
    sy1 = @. 1+0*x1;

    G11 = @. visc * B1 * (rx1 * rx1 + ry1 * ry1);
    G12 = @. visc * B1 * (rx1 * sx1 + ry1 * sy1);
    G22 = @. visc * B1 * (sx1 * sx1 + sy1 * sy1);

    function laplOp(v)
        v = reshape(v,nx1,ny1);
        v = lapl(v,M,[],[],Dr1,Ds1,G11,G12,G22);
        v = reshape(v,nx1*ny1)
        return v;
    end

    rhs = B1 .* f; rhs = reshape(rhs,nx1*ny1);

    return laplOp, rhs
end


# verify solver
# laplOp, rhs = setup(visc, f)
# op = LinearOperator(nx1*ny1,nx1*ny1,true,true,v->laplOp(v),nothing,nothing);
# uloc,stat = Krylov.cg(op,rhs);
# uloc = reshape(uloc,nx1,ny1);
# println(norm(uloc-ut,Inf));
# println(norm(B1.*f - lapl(ut,M,[],[],Dr1,Ds1,Gs...),Inf));

#----------------------------------------------------------------------#
# Adjoint
#----------------------------------------------------------------------#

# Set forward/backwards solver
function solver(lhs, rhs, adj::Bool)
    op = LinearOperator(length(rhs),length(rhs),true,true,v->lhs(v),nothing,nothing)
    if !adj
        return Krylov.cg(op,rhs)[1]
    else
        return Krylov.cg(op',rhs)[1]
    end
end

# Setup problem (returns lhs and rhs)
kx=1
ky=1
ut = sin.(kx*pi*x1).*sin.(ky*pi*y1)
function problem(p)
    visc = @. p[1] + 0*x1;
    f  = p[2] .* ut .* ((kx^2+ky^2)*pi^2);
    return setup(visc, f)
end

# Model and Loss
function model(p)
    u = linsolve(p,problem,solver) # Has adjoint support thru Zygote
    u = reshape(u,nx1,ny1)
end

function loss(p)
    u = model(p)
    l = sum((u.-ut).^2)
    return l, u
end

# Training and plotting
function myplot(u,ut)
    sleep(0.001)
    fig1 = heatmap(u, clim = (minimum(ut),maximum(ut)))
    fig2 = heatmap(ut.-u, clim = (-1e-3,1e-3))
    plot(fig1, fig2, layout=2)
end

callback = function (p, l, pred; doplot = false)
  display(l)
  if doplot
      modifyPlotObject!(fig,arg1=pred)
  end
  return false
end

p0 = [0.1,1.5]
# fig = @makeLivePlot myplot(model(p0),ut)
# sleep(10)
result = DiffEqFlux.sciml_train(loss, p0, LBFGS(), cb = callback, maxiters = 200)
