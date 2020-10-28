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

using .SEM

using FastGaussQuadrature
using Plots, LinearAlgebra, SparseArrays

using LinearOperators
import Krylov

linspace(zi::Number,ze::Number,n::Integer) = range(zi,stop=ze,length=n)
#----------------------------------------------------------------------#
# setup
#----------------------------------------------------------------------#
#ifvl = 1;    # evolve  vel field per NS eqn
#ifad = 1;    # advect  vel, sclr
#ifpr = 1;    # project vel onto a div-free subspace
#ifps = 0;    # evolve sclr per advection diffusion eqn

nx1 = 32; Ex = 1;
ny1 = 32; Ey = 1;

nx2 = nx1-2; nxd = Int(ceil(1.5*nx1)); nxo = 10*nx1;
ny2 = nx1-2; nyd = Int(ceil(1.5*ny1)); nyo = 10*ny1;
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

# deform grid with gordonhall

Jac1,Jaci1,rx1,ry1,sx1,sy1 = jac(x1,y1,Dx1,Dy1);
Jac2,Jaci2,rx2,ry2,sx2,sy2 = jac(x2,y2,Dx2,Dy2);
Jacd,Jacid,rxd,ryd,sxd,syd = jac(xd,yd,Dxd,Dyd);

#----------------------------------------------------------------------#
# case setup
#----------------------------------------------------------------------#
# prescribe forcing, true solution, boundary data
kx=1.
ky=1.
ut = @. sin(kx*pi*x1)*sin.(ky*pi*y1) # true solution
f  = @. ut*((kx^2+ky^2)*pi^2);       # forcing/RHS
ub = copy(ut);                       # boundary data

visc = @. 1+0*x1;

f = @. 1+0*x1;
ub= @. 0+0*x1;
#----------------------------------------------------------------------#
# setup
#----------------------------------------------------------------------#

# diagonal mass matrix
B1  = Jac1 .* (wx1*wy1');
B2  = Jac2 .* (wx2*wy2');
Bd  = Jacd .* (wxd*wyd');
Bi1 = 1 ./ B1;

# # set up matrices for Laplace op.
# G11 = @. visc * B1 * (rx1 * rx1 + ry1 * ry1);
# G12 = @. visc * B1 * (rx1 * sx1 + ry1 * sy1);
# G22 = @. visc * B1 * (sx1 * sx1 + sy1 * sy1);
#
# #----------------------------------------------------------------------#
# # solve
# #----------------------------------------------------------------------#
# # CG solver misbehaving:
# # - too many iterations on the eigen problem - often not converging
# # - Laplacian operator not hermitian for #Elem > 1.
# #
# #----------------------------------------------------------------------#
# # Laplace Fast Diagonalization Preconditioner
# #----------------------------------------------------------------------#
# rx = rx1[1];
# sy = sy1[1];
# Ja = Jac1[1];
# Bx = Matrix(Diagonal(wr1 ./ rx));
# By = Matrix(Diagonal(ws1 ./ rx));
# Dx = rx*Dr1;
# Dy = sy*Ds1;
# Ax = Dx'*Bx*Dx;
# Ay = Dy'*By*Dy;
# #
# #Br = Rx*Br*Rx';
# #Bs = Ry*Bs*Ry';
# #Ar = Rx*Ar*Rx';
# #As = Ry*As*Ry';
#
# Lx,Sx = eigen(Ax,Bx); Sxi = inv(Sx);
# Ly,Sy = eigen(Ay,By); Syi = inv(Sy);
# ox=ones(size(Lx));
# oy=ones(size(Ly));
# Lfdm  = Lx*oy' + ox*Ly';
# Lfdmi = 1 ./ Lfdm;
# for j=1:ny1 for i=1:nx1
#     if(abs(Lfdmi[i,j])>1e8) Lfdmi[i,j] = 0; end
# end end
# Sx  = kron(Iex,Sx ); Sy  = kron(Iey,Sy );
# Sxi = kron(Iex,Sxi); Syi = kron(Iey,Syi);
# Lfdmi = kron(ones(Ex,Ey),Lfdmi);
#
# Li = Diagonal(reshape(Lfdmi,:));
# Bi = Diagonal(reshape(Bi1  ,:));
# FD = kron(Sy,Sx) * Li * kron(Syi,Sxi) * Bi;
# #----------------------------------------------------------------------#
# # Explicit system
# #----------------------------------------------------------------------#
# # restriction on global vector
# Ix1 = Matrix(I,Ex*(nx1-1)+1,Ex*(nx1-1)+1); Rx1 = Ix1[2:end-1,:];
# Iy1 = Matrix(I,Ey*(ny1-1)+1,Ey*(ny1-1)+1); Ry1 = Iy1[2:end-1,:];
# R   = kron(Ry1,Rx1);
# Q   = kron(Qy1,Qx1);
# Ix1 = Matrix(I,Ex*nx1,Ex*nx1); # deriv mats on local vector
# Iy1 = Matrix(I,Ey*ny1,Ey*ny1);
# Dr  = kron(Iy1,Dx1);
# Ds  = kron(Dy1,Ix1);
# Drs = [Dr;Ds];
#
# B   = Diagonal(reshape(B1 ,:));
# g11 = Diagonal(reshape(G11,:));
# g12 = Diagonal(reshape(G12,:));
# g22 = Diagonal(reshape(G22,:));
# G   = [g11 g12; g12 g22];
# A   = Drs' * G * Drs;
#
# # full rank system acting on global, restricted vectors
# AA  = R * Q' * A * Q * R';
# BB  = R * Q' * B * Q * R';
# Qf  = reshape(f,:);
# bb  = R * Q' * B * Qf;
# uu ,stuu = Krylov.cg(AA,bb);
# #uu = pcg(bb,AA,[]);
# uu = Q * R' * uu;
# uu = reshape(uu,Ex*nx1,Ey*ny1);
#
# # Restriction on local vector
# Ix1 = Matrix(I,Ex*nx1,Ex*nx1);
# Iy1 = Matrix(I,Ey*ny1,Ey*ny1);
# Rx1 = Ix1[2:end-1,:];
# Ry1 = Iy1[2:end-1,:];
# R   = kron(Ry1,Rx1);
# M   = R'*R;
# QQt = Q*Q';
#
# # rank deficient system acting on local, unrestricted operators
# Aloc = QQt * M * A * M;
# Bloc = QQt * M * B * M;
# bloc = QQt * M * B * Qf;
# uloc,stloc = Krylov.cgls(Aloc,bloc);
# #uloc = pcg(bloc,Aloc,[]);
# uloc = reshape(uloc,Ex*nx1,Ey*ny1);
#
# #----------------------------------------------------------------------#
# function laplOp(v)
#     v = reshape(v,nx1*Ex,ny1*Ey)
#     w = lapl(v,M1,Qx1,Qy1,Dx1,Dy1,G11,G12,G22);
#     w = reshape(w,nx1*ny1*Ex*Ey);
#     return w
# end
# function laplFdmOp(v)
#     v = reshape(v,nx1*Ex,ny1*Ey)
#     w = lapl_fdm(v,Bi1,Sx,Sy,Sxi,Syi,Lfdmi);
#     w = reshape(w,nx1*ny1*Ex*Ey);
#     return w
# end
# opLapl = LinearOperator(nx1*ny1*Ex*Ey,nx1*ny1*Ex*Ey,true,true
#                         ,v->laplOp(v),nothing,nothing)
#
# opLaplFdm = LinearOperator(nx1*ny1*Ex*Ey,nx1*ny1*Ex*Ey,true,true
#                         ,v->laplFdmOp(v),nothing,nothing)
#
# b   = mass(f,[],B1,[],[]);
# #b .-= lapl(ub,[],[],[],Dx1,Dy1,G11,G12,G22);
# b  .= mass(b,M1,[],Qx1,Qy1);
#
# r  = reshape(b,nx1*ny1*Ex*Ey);
# u,st = Krylov.cg(opLapl,r);
# u = reshape(u,nx1*Ex,ny1*Ey);
# #u = u + ub;
# #norm(b,Inf),norm(ut-u,Inf)

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

    b   = mass(f,[],B1,[],[])
    b   = mass(b,M1,[],Qx1,Qy1)

    rhs  = reshape(b,nx1*ny1*Ex*Ey);

    return laplOp, rhs
end

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
V = 4
p = 5
ε = 1e-3
α = 1e2
f1 = 1e-2 .+ 0*x1
k(a) = @. ε + (1-ε)*a^p
kx=1.
ky=1.
ut = @. sin(kx*pi*x1)*sin.(ky*pi*y1) # true solution
M1[1,:] .= 1; M1[:,end] .= 1
function problem(p)
    visc = k(p)
    f  = f1
    return setup(visc, f)
end

# Model and Loss
function model(p)
    u = linsolve(p,problem,solver) # Has adjoint support thru Zygote
    u = reshape(u,nx1*Ex,ny1*Ey)
end

function loss(a)
    u = model(a)
    adx = Dr1*a
    ady = Ds1*a
    l = sum(B1.*f1.*u)
    l += α*sum(B1.*adx.*adx)
    l += α*sum(B1.*ady.*ady)
    # l = 0.5*sum(abs2,u.-ut)
    return l, u
end

function fmin(a::Vector, grad::Vector)
    a = reshape(a,nx1*Ex,ny1*Ey)
    if length(grad)>0
        grad[:] = gradient((a)->loss(a)[1], a)[1][:]
    end
    return loss(a)[1]
end
function fc(a::Vector, grad::Vector)
    a = reshape(a,nx1*Ex,ny1*Ey)
    ineq(a) = sum(B1.*a) .- V
    if length(grad)>0
        grad[:] = gradient((a)->ineq(a), a)[1][:]
    end
    return ineq(a)
end

# Training and plotting
global param = []
callback = function (p, l, pred; doplot = false)
  display(l)
  global param = p
  if doplot
      modifyPlotObject!(fig,arg1=pred)
  end
  return false
end

p0 = rand(nx1*Ex,ny1*Ey)
# fig = @makeLivePlot myplot(model(p0),ut)
# sleep(10)
opt = NLopt.Opt(:LD_MMA, length(p0))
opt.min_objective = fmin
opt.lower_bounds = 0.
opt.upper_bounds = 1.
# inequality_constraint!(opt, fc)
opt.maxtime = 60
opt.xtol_abs = 1e-6
opt.xtol_rel = 1e-4
#(optf,optx,ret) = NLopt.optimize(opt, p0[:])
result = DiffEqFlux.sciml_train(loss, p0, Fminbox(GradientDescent()),
                                lower_bounds=0, upper_bounds=1, allow_f_increases = true,
                                cb = callback, maxiters = 200)
#heatmap(model(reshape(optx,nx1,ny1)))
#heatmap(reshape(optx,nx1*Ex,ny1*Ey))
