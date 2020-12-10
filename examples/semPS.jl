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
using DiffEqFlux, Optim
using Flux
using SmoothLivePlot
using Statistics

#----------------------------------------------------------------------#
# setup
#----------------------------------------------------------------------#
nx1 = 8; Ex = 5;
ny1 = 8; Ey = 5;

#nx2 = nx1-2; nxd = Int(ceil(1.5*nx1)); nxo = 10*nx1;
#ny2 = nx1-2; nyd = Int(ceil(1.5*ny1)); nyo = 10*ny1;

#----------------------------------------------------------------------#
# nodal operators
#----------------------------------------------------------------------#
zr1,wr1 = gausslobatto(nx1); zs1,ws1 = gausslobatto(ny1);
#zr2,wr2 = gausslobatto(nx2); zs2,ws2 = gausslobatto(ny2);
#zrd,wrd = gausslobatto(nxd); zsd,wsd = gausslobatto(nyd);
#zro     =linspace(-1,1,nxo); zso     =linspace(-1,1,nyo);

#Jr1d = interpMat(zrd,zr1); Js1d = interpMat(zsd,zs1);
#Jr2d = interpMat(zrd,zr2); Js2d = interpMat(zsd,zs2);
#Jr21 = interpMat(zr1,zr2); Js21 = interpMat(zs1,zs2);
#Jr1o = interpMat(zro,zr1); Js1o = interpMat(zso,zs1);

Dr1 = derivMat(zr1); Ds1 = derivMat(zs1);
#Dr2 = derivMat(zr2); Ds2 = derivMat(zs2);
#Drd = derivMat(zrd); Dsd = derivMat(zsd);

#----------------------------------------------------------------------#
# local operators
#----------------------------------------------------------------------#
wx1 = kron(ones(Ex,1),wr1); wy1 = kron(ones(Ey,1),ws1);
#x2 = kron(ones(Ex,1),wr2); wy2 = kron(ones(Ey,1),ws2);
#xd = kron(ones(Ex,1),wrd); wyd = kron(ones(Ey,1),wsd);

Iex = Matrix(I,Ex,Ex);
Iey = Matrix(I,Ey,Ey);

Dx1 = kron(Iex,Dr1); Dy1 = kron(Iey,Ds1);
#x2 = kron(Iex,Dr2); Dy2 = kron(Iey,Ds2);
#xd = kron(Iex,Drd); Dyd = kron(Iey,Dsd);

#x1d = kron(Iex,Jr1d); Jy1d = kron(Iey,Js1d);
#x21 = kron(Iex,Jr21); Jy21 = kron(Iey,Js21);

#----------------------------------------------------------------------#
# boundary conditions
#----------------------------------------------------------------------#
ifprdcX = false
ifprdcY = false

Ix1 = Matrix(I,Ex*nx1,Ex*nx1);
Iy1 = Matrix(I,Ey*ny1,Ey*ny1);

Rx1 = Ix1[2:end-1,:];
Ry1 = Iy1[2:end-1,:];

if(ifprdcX) Rx1 = Ix1; end
if(ifprdcY) Ry1 = Iy1; end

M1 = diag(Rx1'*Rx1) * diag(Ry1'*Ry1)';

Ix1 = nothing
Iy1 = nothing
Rx1 = nothing
Ry1 = nothing
#----------------------------------------------------------------------#
# mapping
#----------------------------------------------------------------------#

# Q: global -> local op, Q': local -> global
Qx1 = semq(Ex,nx1,ifprdcX); Qy1 = semq(Ey,ny1,ifprdcY);
#Qx2 = semq(Ex,nx2,ifprdcX); Qy2 = semq(Ey,ny2,ifprdcY);

# gather scatter op
QQtx1 = Qx1*Qx1';
QQty1 = Qy1*Qy1';

#Qtx2 = Qx2*Qx2';
#Qty2 = Qy2*Qy2';

# weight for inner products
mult = ones(size(M1));
mult = ABu(QQty1,QQtx1,mult);
mult = @. 1 / mult;
#----------------------------------------------------------------------#
# geometry
#----------------------------------------------------------------------#
x1e,_ = semmesh(Ex,nx1); y1e,_ = semmesh(Ey,ny1);
#2e,_ = semmesh(Ex,nx2); y2e,_ = semmesh(Ey,ny2);
#de,_ = semmesh(Ex,nxd); yde,_ = semmesh(Ey,nyd);

x1,y1 = ndgrid(x1e,y1e);
#2,y2 = ndgrid(x2e,y2e);
#d,yd = ndgrid(xde,yde);

# x1 = @. 0.5 * (x1 + 1); y1 = @. 0.5 * (y1 + 1);
# x2 = @. 0.5 * (x2 + 1); y2 = @. 0.5 * (y2 + 1);
# xd = @. 0.5 * (xd + 1); yd = @. 0.5 * (yd + 1);

# deform grid with gordonhall

Jac1,Jaci1,rx1,ry1,sx1,sy1 = jac(x1,y1,Dx1,Dy1);
#ac2,Jaci2,rx2,ry2,sx2,sy2 = jac(x2,y2,Dx2,Dy2);
#acd,Jacid,rxd,ryd,sxd,syd = jac(xd,yd,Dxd,Dyd);

# diagonal mass matrix
B1  = Jac1 .* (wx1*wy1');
#2  = Jac2 .* (wx2*wy2');
#d  = Jacd .* (wxd*wyd');
Bi1 = 1 ./ B1;

#----------------------------------------------------------------------#
# setup
#----------------------------------------------------------------------#
function setup(visc, f)

    G11 = @. visc * B1 * (rx1 * rx1 + ry1 * ry1)
    G12 = @. visc * B1 * (rx1 * sx1 + ry1 * sy1)
    G22 = @. visc * B1 * (sx1 * sx1 + sy1 * sy1)

    function laplOp(v)
        w = lapl(v,M1,Qx1,Qy1,Dx1,Dy1,G11,G12,G22);
        return w
    end

    b   = mass(f,[],B1,[],[])
    b   = mass(b,M1,[],Qx1,Qy1)
    rhs = b;

    return laplOp, rhs
end

function solver(opA,rhs,adj::Bool)
    if !adj return pcg(rhs,opA,mult)
    else    return pcg(rhs,opA,mult)
    end
end
#----------------------------------------------------------------------#
# test solver
#----------------------------------------------------------------------#
kx   = 1.;
ky   = 1.;
ut   = @. sin(kx*pi*x1)*sin(ky*pi*y1); # true solution
f    = @. ut*((kx^2+ky^2)*pi^2);       # forcing/RHS
ub   = copy(ut);                       # boundary data
visc = @. 1+0*x1;

op,r = setup(visc,f);
u    = solver(op,r,false);
er   = (norm(u-ut,Inf));
println("solver working fine, residual: ",er);
#----------------------------------------------------------------------#
# set up case
#----------------------------------------------------------------------#

V = 0.4
p = 5
ε = 1e-3
α = 1e-8
f1 = 1e-2 .+ 0*x1
kond(a) = @. ε + (1-ε)*a^p
#M1[:,end] .= 1; M1[1,:] .= 1

function problem(p)
    visc = kond(p)
    f    = f1
    return setup(visc, f)
end

function model(p)
    u = linsolve(p,problem,solver) # Has adjoint support thru Zygote
end

at = y1.^2
at = @. 0.5 + x1*0
ut = model(at)
af(p) = @. 0.5*(tanh(p)+1)
function loss(p)
    a = af(p)
    u = model(a)
    #adx,ady = grad(a,Dr1,Ds1,rx1,ry1,sx1,sy1);
    #vv = @. f1*u + α*(adx^2+ady^2);
    #l  = sum(B1.*vv);
    ##debugging
    e = @. u - ut;
    e = @. a - 0.5;
    n = length(e);
    l = (sum(e.^2)/n);
    return l, u
end

#----------------------------------------------------------------------#
# DiffEqFlux.sciml_train
#----------------------------------------------------------------------#
dp = false
p0 = @. 1 + 0*x1
#p0 = rand(nx1*Ex,ny1*Ey)

function myplot(u,ut)
    #sleep(0.001)
    plot(mesh(x1,y1,ut),mesh(x1,y1,u))
end
if dp
    fig = @makeLivePlot myplot(model(af(p0)),ut)
    sleep(5)
end

global param = []
callback = function (p, l, pred; doplot = dp)
  #display(l)
  global param = p
  if doplot
      modifyPlotObject!(fig,arg1=pred)
  end
  return false
end

result = DiffEqFlux.sciml_train(loss,p0,ADAM(1e-2),cb=Flux.throttle(callback,1),maxiters=100)
display(loss(param)[1])

p=param;dp=Zygote.gradient((p)->loss(p)[1], p)[1];

plot(mesh(x1,y1,ut),mesh(x1,y1,model(af(param))),
     mesh(x1,y1,at),mesh(x1,y1,af(param)))
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
