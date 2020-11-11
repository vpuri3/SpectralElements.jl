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
import Krylov

import  Zygote

import NLopt
import DiffEqFlux, Optim
import Flux

#linspace(zi::Number,ze::Number,n::Integer) = range(zi,stop=ze,length=n)
#----------------------------------------------------------------------#
# setup
#----------------------------------------------------------------------#
#ifvl = 1;    # evolve  vel field per NS eqn
#ifad = 1;    # advect  vel, sclr
#ifpr = 1;    # project vel onto a div-free subspace
#ifps = 0;    # evolve sclr per advection diffusion eqn

nx1 = 16; Ex = 1;
ny1 = 16; Ey = 1;

nx2 = nx1-2; nxd = ceil(1.5*nx1); nxo = 10*nx1;
ny2 = nx1-2; nyd = ceil(1.5*ny1); nyo = 10*ny1;
#----------------------------------------------------------------------#
# nodal operators
#----------------------------------------------------------------------#
zr1,wr1 = gausslobatto(nx1); zs1,ws1 = gausslobatto(ny1);
zr2,wr2 = gausslobatto(nx2); zs2,ws2 = gausslobatto(ny2);
zrd,wrd = gausslobatto(nxd); zsd,wsd = gausslobatto(nyd);
zro     =linspace(-1,1,nxo); zso     =linspace(-1,1,nyo);

Jr1d = interpMat(zrd,zr1); Js1d = interpMat(zsd,zs1);
Jr2d = interpMat(zrd,zr2); Js2d = interpMat(zsd,zs2);
Jr21 = interpMat(zr1,zr2); Js21 = interpMat(zs1,zs2);
Jr1o = interpMat(zro,zr1); Js1o = interpMat(zso,zs1);

Dr1 = derivMat(zr1); Ds1 = derivMat(zs1);
Dr2 = derivMat(zr2); Ds2 = derivMat(zs2);
Drd = derivMat(zrd); Dsd = derivMat(zsd);

#----------------------------------------------------------------------#
# local operators
#----------------------------------------------------------------------#
wx1 = kron(ones(Ex,1),wr1); wy1 = kron(ones(Ey,1),ws1);
wx2 = kron(ones(Ex,1),wr2); wy2 = kron(ones(Ey,1),ws2);
wxd = kron(ones(Ex,1),wrd); wyd = kron(ones(Ey,1),wsd);

Iex = Matrix(I,Ex,Ex);
Iey = Matrix(I,Ey,Ey);

Dx1 = kron(Iex,Dr1); Dy1 = kron(Iey,Ds1);
Dx2 = kron(Iex,Dr2); Dy2 = kron(Iey,Ds2);
Dxd = kron(Iex,Drd); Dyd = kron(Iey,Dsd);

Jx1d = kron(Iex,Jr1d); Jy1d = kron(Iey,Js1d);
Jx21 = kron(Iex,Jr21); Jy21 = kron(Iey,Js21);

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
Qx2 = semq(Ex,nx2,ifprdcX); Qy2 = semq(Ey,ny2,ifprdcY);

# gather scatter op
QQtx1 = Qx1*Qx1';
QQty1 = Qy1*Qy1';

QQtx2 = Qx2*Qx2';
QQty2 = Qy2*Qy2';

#----------------------------------------------------------------------#
# geometry
#----------------------------------------------------------------------#
x1e,_ = semmesh(Ex,nx1); y1e,_ = semmesh(Ey,ny1);
x2e,_ = semmesh(Ex,nx2); y2e,_ = semmesh(Ey,ny2);
xde,_ = semmesh(Ex,nxd); yde,_ = semmesh(Ey,nyd);

x1,y1 = ndgrid(x1e,y1e);
x2,y2 = ndgrid(x2e,y2e);
xd,yd = ndgrid(xde,yde);

x1 = @. 0.5 * (x1 + 1); y1 = @. 0.5 * (y1 + 1);
x2 = @. 0.5 * (x2 + 1); y2 = @. 0.5 * (y2 + 1);
xd = @. 0.5 * (xd + 1); yd = @. 0.5 * (yd + 1);

# deform grid with gordonhall

Jac1,Jaci1,rx1,ry1,sx1,sy1 = jac(x1,y1,Dx1,Dy1);
Jac2,Jaci2,rx2,ry2,sx2,sy2 = jac(x2,y2,Dx2,Dy2);
Jacd,Jacid,rxd,ryd,sxd,syd = jac(xd,yd,Dxd,Dyd);

# diagonal mass matrix
B1  = Jac1 .* (wx1*wy1');
B2  = Jac2 .* (wx2*wy2');
Bd  = Jacd .* (wxd*wyd');
Bi1 = 1 ./ B1;

#----------------------------------------------------------------------#
# setup
#----------------------------------------------------------------------#

function setup(visc, f)

    G11 = @. visc * B1 * (rx1 * rx1 + ry1 * ry1)
    G12 = @. visc * B1 * (rx1 * sx1 + ry1 * sy1)
    G22 = @. visc * B1 * (sx1 * sx1 + sy1 * sy1)

    function laplOp(v)
        v = reshape(v,nx1*Ex,ny1*Ey)
        w = lapl(v,M1,Qx1,Qy1,Dx1,Dy1,G11,G12,G22);
        w = reshape(w,nx1*ny1*Ex*Ey);
        return w
    end

    b = mass(f,[],B1,[],[])
    b = mass(b,M1,[],Qx1,Qy1)

    rhs  = reshape(b,nx1*ny1*Ex*Ey);

    return laplOp, rhs
end

function solver(lhs, rhs, adj::Bool)
    op = LinearOperator(length(rhs),length(rhs),true,true,
                        v->lhs(v),nothing,nothing)
    if !adj
        return Krylov.cg(op,rhs)[1]
    else
        return Krylov.cg(op',rhs)[1]
    end
end

function model(p)
    u = linsolve(p,problem,solver) # Has adjoint support thru Zygote
    u = reshape(u,nx1*Ex,ny1*Ey)
end

function loss(a)
    u = model(a)
    adx,ady = grad(a,Dr1,Ds1,rx1,ry1,sx1,sy1);
    vv  = @. f1*u + α*(adx^2+ady^2);
    l   = sum(B1.*vv);
    return l,u
end

#----------------------------------------------------------------------#
# test solver
#----------------------------------------------------------------------#
kx   = 1.
ky   = 1.
ut   = @. sin(kx*pi*x1)*sin(ky*pi*y1) # true solution
f    = @. ut*((kx^2+ky^2)*pi^2);      # forcing/RHS
ub   = copy(ut);                      # boundary data
visc = @. 1+0*x1;

op,r = setup(visc,f)
u = solver(op,r,false)
u = reshape(u,nx1*Ex,ny1*Ey)
nrm = (norm(u-ut,Inf))
println("solver working fine, residual: ",nrm)
#----------------------------------------------------------------------#
# set up case
#----------------------------------------------------------------------#
V = 0.4
p = 5
ε = 1e-3
α = 1e-4
f1 = 1e-2 .+ 0*x1
k(a) = @. ε + (1-ε)*a^p
M1[1,:]   .= 1;
M1[:,end] .= 1;
function problem(p)
    visc = k(p)
    f  = f1
    return setup(visc, f)
end

#----------------------------------------------------------------------#
# NLopt.Optimize
#----------------------------------------------------------------------#
function fmin(a::Vector, grad::Vector)
    a = reshape(a,nx1*Ex,ny1*Ey)
    if length(grad)>0
        grad[:] = Zygote.gradient((a)->loss(a)[1], a)[1][:]
    end
    return loss(a)[1]
end

function fc(a::Vector, grad::Vector)
    a = reshape(a,nx1*Ex,ny1*Ey)
    ineq(a) = sum(B1.*a) .- V
    if length(grad)>0
        grad[:] = Zygote.gradient((a)->ineq(a), a)[1][:]
    end
    return ineq(a)
end

# set up and run optimizer
a0 = rand(nx1*Ex,ny1*Ey)
a0 = 0.4*ones(nx1*Ex,ny1*Ey)

fig = mesh(x1,y1,model(a0)); display(fig);

opt = NLopt.Opt(:LD_MMA, length(a0))
opt.min_objective = fmin
NLopt.inequality_constraint!(opt, fc)
opt.lower_bounds = 0.
opt.upper_bounds = 1.
opt.maxeval = 20
opt.xtol_abs = 1e-8
opt.xtol_rel = 1e-8

#(optf,optx,ret) = NLopt.optimize(opt, a0[:])

#opta = reshape(optx,nx1*Ex,ny1*Ey)
#numevals = opt.numevals
#println("started with loss ",loss(a0)[1])
#println("got $optf after $numevals iterations (returned $ret)")
#fig = mesh(x1,y1,model(opta)); display(fig);
#----------------------------------------------------------------------#
# DiffEqFlux.sciml_train
#----------------------------------------------------------------------#
p0 = rand(nx1*Ex,ny1*Ey)
p0 = 0.4*ones(nx1*Ex,ny1*Ey)

global param = []
callback = function (p, l, pred; doplot = false)
  display(l)
  global param = p
  if doplot
      modifyPlotObject!(fig,arg1=pred)
  end
  return false
end

result = DiffEqFlux.sciml_train(loss,p0,Optim.Fminbox(Optim.GradientDescent()),
        lower_bounds=0, upper_bounds=1, allow_f_increases = true,
        cb = callback, maxiters = 20)

#result = DiffEqFlux.sciml_train(loss, p0, Flux.ADAM(1e-1),cb = callback, maxiters = 20)
#a = @. 0.5*(tanh(p)+1)
a=0 .*x1.+0.4;da=Zygote.gradient((a)->loss(a)[1], a)[1]; mesh(x1,y1,da)
#----------------------------------------------------------------------#
