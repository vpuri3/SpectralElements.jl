#!/usr/bin/env julia

#struct session
#    ifvl::Bool
#    ifad::Bool
#    ifpr::Bool
#    ifps::Bool
#
#    nx1::Int
#    ny1::Int
#end

using SEM

using FastGaussQuadrature
using Plots, LinearAlgebra, SparseArrays

using LinearOperators
using Krylov

using  Zygote

using NLopt
using DiffEqFlux,Optim
using Flux
using Statistics

#----------------------------------------------------------------------#
# size
#----------------------------------------------------------------------#
nx1 = 12; Ex = 8;
ny1 = 12; Ey = 8;

nx2 = nx1-2; nxd = Int(ceil(1.5*nx1));
ny2 = nx1-2; nyd = Int(ceil(1.5*ny1));

nxp = 3*nx1;
nyp = 3*ny1;

#----------------------------------------------------------------------#
# nodal operators
#----------------------------------------------------------------------#
zr1,wr1 = gausslobatto(nx1);  zs1,ws1 = gausslobatto(ny1);
zr2,wr2 = gausslobatto(nx2);  zs2,ws2 = gausslobatto(ny2);
zrd,wrd = gausslobatto(nxd);  zsd,wsd = gausslobatto(nyd);
zrp     = linspace(-1,1,nxp); zsp     = linspace(-1,1,nyp);

Jr1d = interpMat(zrd,zr1); Js1d = interpMat(zsd,zs1);
Jr2d = interpMat(zrd,zr2); Js2d = interpMat(zsd,zs2);
Jr21 = interpMat(zr1,zr2); Js21 = interpMat(zs1,zs2);
Jr1p = interpMat(zrp,zr1); Js1p = interpMat(zsp,zs1);

Dr1 = derivMat(zr1); Ds1 = derivMat(zs1);
Dr2 = derivMat(zr2); Ds2 = derivMat(zs2);
Drd = derivMat(zrd); Dsd = derivMat(zsd);

#----------------------------------------------------------------------#
# boundary conditions
#----------------------------------------------------------------------#
ifperiodicX = false
ifperiodicY = false

Ix1 = Matrix(I,Ex*nx1,Ex*nx1);
Iy1 = Matrix(I,Ey*ny1,Ey*ny1);

Rx1 = Ix1[2:end-1,:];
Ry1 = Iy1[2:end-1,:];

if(ifperiodicX) Rx1 = Ix1; end
if(ifperiodicY) Ry1 = Iy1; end

M1 = diag(Rx1'*Rx1) * diag(Ry1'*Ry1)'; # hom. dir BC

Ix1 = nothing
Iy1 = nothing
Rx1 = nothing
Ry1 = nothing
#----------------------------------------------------------------------#
# mapping
#----------------------------------------------------------------------#

# Q: global -> local op, Q': local -> global
Qx1 = semq(Ex,nx1,ifperiodicX); Qx2 = semq(Ex,nx2,ifperiodicX);
Qy1 = semq(Ey,ny1,ifperiodicY); Qy2 = semq(Ey,ny2,ifperiodicY);

# gather scatter op
QQtx1 = Qx1*Qx1'; QQtx2 = Qx2*Qx2';
QQty1 = Qy1*Qy1'; QQty2 = Qy2*Qy2';

# weight for inner products
mult1 = ones(nx1*Ex,ny1*Ey);
mult1 = gatherScatter(mult1,QQtx1,QQty1);
mult1 = @. 1 / mult1;

mult2 = ones(nx2*Ex,ny2*Ey);
mult2 = gatherScatter(mult2,QQtx2,QQty2);
mult2 = @. 1 / mult2;
#----------------------------------------------------------------------#
# geometry
#----------------------------------------------------------------------#
x1e,_ = semmesh(Ex,nx1); y1e,_ = semmesh(Ey,ny1);
x2e,_ = semmesh(Ex,nx2); y2e,_ = semmesh(Ey,ny2);
xde,_ = semmesh(Ex,nxd); yde,_ = semmesh(Ey,nyd);

x1,y1 = ndgrid(x1e,y1e);
x2,y2 = ndgrid(x2e,y2e);
xd,yd = ndgrid(xde,yde);

# deform grid with gordonhall
x1 = @. 0.5 * (x1 + 1); y1 = @. 0.5 * (y1 + 1);
x2 = @. 0.5 * (x2 + 1); y2 = @. 0.5 * (y2 + 1);
xd = @. 0.5 * (xd + 1); yd = @. 0.5 * (yd + 1);

Jac1,Jaci1,rx1,ry1,sx1,sy1 = jac(x1,y1,Dr1,Ds1);
Jac2,Jaci2,rx2,ry2,sx2,sy2 = jac(x2,y2,Dr2,Ds2);
Jacd,Jacid,rxd,ryd,sxd,syd = jac(xd,yd,Drd,Dsd);

wx1 = kron(ones(Ex,1),wr1); wy1 = kron(ones(Ey,1),ws1);
wx2 = kron(ones(Ex,1),wr2); wy2 = kron(ones(Ey,1),ws2);
wxd = kron(ones(Ex,1),wrd); wyd = kron(ones(Ey,1),wsd);

# diagonal mass matrix
B1  = Jac1 .* (wx1*wy1');
B2  = Jac2 .* (wx2*wy2');
Bd  = Jacd .* (wxd*wyd');
Bi1 = 1 ./ B1;

# don't use lumped mass matrices for now
#Jr1d = []
#Js1d = []
#rxd = rx1
#ryd = ry1
#sxd = sx1
#syd = sy1
#Bd = B1

#----------------------------------------------------------------------#
# set up case
#----------------------------------------------------------------------#
V  = 0.4
pa = 5
ε  = 1e-3
α  = 1e-8
f1 = @. 1e-2 + 0*x1
M1[:,end] .= 1;
M1[1,:]   .= 1;
af(p)   = @. 0.5*(tanh(p)+1)
kond(a) = @. ε + (1-ε)*a^pa

function problem(a) # topology parameter, forcing

    Ba = Bd;
    #Ba = Bd .* ABu(Js1d,Jr1d,a);

    visc  = kond(a)
    viscd = ABu(Js1d,Jr1d,visc);
    G11 = @. viscd * Ba * (rxd * rxd + ryd * ryd);
    G12 = @. viscd * Ba * (rxd * sxd + ryd * syd);
    G22 = @. viscd * Ba * (sxd * sxd + syd * syd);

    lhs(v) = lapl(v,M1,Jr1d,Js1d,QQtx1,QQty1,Dr1,Ds1,G11,G12,G22);

    f   = f1;
    rhs = mass(f,M1,Ba,Jr1d,Js1d,QQtx1,QQty1);

    return lhs, rhs
end

function opM(v) # preconditioner
    return v
end

function solver(opA,rhs,adj::Bool)
    if adj
        rhs = mass(rhs,M1,mult1,[],[],QQtx1,QQty1);
        return pcg(rhs,opA,opM,mult1,false);
    else
        return pcg(rhs,opA,opM,mult1,false);
    end
end

function model(a)
    u = linsolve(a,problem,solver) # adjoint support thru Zygote
end

pt = @. 0.5*(1+1*sin(2*pi*x1)*sin(2*pi*y1));
pt = @. 0.1 + 0.5*(x1+y1);
at = af(pt);
ut = model(at);
function loss(p)
    a = af(p)
    u = model(a)
    adx,ady = grad(a,Dr1,Ds1,rx1,ry1,sx1,sy1);
    adx = mass(adx,[],mult1,[],[],QQtx1,QQty1);
    ady = mass(ady,[],mult1,[],[],QQtx1,QQty1);
    ll = @. f1*u + α*(adx^2+ady^2);
    l  = sum(B1.* a.*ll);
    #e = @. u - ut;
    #l = sum(B1.*e.*e);
    return l, u
end

#----------------------------------------------------------------------#
# DiffEqFlux.sciml_train
#----------------------------------------------------------------------#
p0 = @. 0.6 + 0.0*x1
p0 = mult1 .* gatherScatter(p0,QQtx1,QQty1);

global param = []
callback = function (p, l, pred; doplot = true)
  global param = p
  return false
end

result = DiffEqFlux.sciml_train(loss,p0,ADAM(5e-3),cb=Flux.throttle(callback,1),maxiters=10)

p=param;dp=Zygote.gradient((p)->loss(p)[1], p)[1];
Jp = ABu(Js1p,Jr1p,p);
pl=heatmap(af(Jp))
#pl=plot(mesh(x1,y1,ut),mesh(x1,y1,model(af(p))),
#        mesh(x1,y1,at),mesh(x1,y1,af(p)))
display(pl),display(loss(param)[1])
#----------------------------------------------------------------------#
# NLopt.Optimize
#----------------------------------------------------------------------#
#function fmin(a::Vector, grad::Vector)
#    a = reshape(a,nx1*Ex,ny1*Ey)
#    if length(grad)>0
#        grad[:] = Zygote.gradient((a)->loss(a)[1], a)[1][:]
#    end
#    return loss(a)[1]
#end

#function fc(a::Vector, grad::Vector)
#    a = reshape(a,nx1*Ex,ny1*Ey)
#    ineq(a) = sum(B1.*a) .- V
#    if length(grad)>0
#        grad[:] = Zygote.gradient((a)->ineq(a), a)[1][:]
#    end
#    return ineq(a)
#end

## set up and run optimizer
#a0 = rand(nx1*Ex,ny1*Ey)
#a0 = 0.4*ones(nx1*Ex,ny1*Ey)

#fig = mesh(x1,y1,model(a0)); display(fig);

#opt = NLopt.Opt(:LD_MMA, length(a0))
#opt.min_objective = fmin
#NLopt.inequality_constraint!(opt, fc)
#opt.lower_bounds = 0.
#opt.upper_bounds = 1.
#opt.maxeval = 20
#opt.xtol_abs = 1e-8
#opt.xtol_rel = 1e-8

#(optf,optx,ret) = NLopt.optimize(opt, a0[:])

#opta = reshape(optx,nx1*Ex,ny1*Ey)
#numevals = opt.numevals
#println("started with loss ",loss(a0)[1])
#println("got $optf after $numevals iterations (returned $ret)")
#fig = mesh(x1,y1,model(opta)); display(fig);
#----------------------------------------------------------------------#
nothing
