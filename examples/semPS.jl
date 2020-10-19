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

#----------------------------------------------------------------------#
# setup
#----------------------------------------------------------------------#
ifvl = 1;    # evolve  vel field per NS eqn
ifad = 1;    # advect  vel, sclr
ifpr = 1;    # project vel onto a div-free subspace
ifps = 0;    # evolve sclr per advection diffusion eqn

nx1 = 8; Ex = 4;
ny1 = 8; Ey = 4;

nx2 = nx1-2; nxd = Int(ceil(1.5*nx1)); nxo = 10*nx1; 
ny2 = nx1-2; nyd = Int(ceil(1.5*ny1)); nyo = 10*ny1; 
#----------------------------------------------------------------------#
# nodal operators
#----------------------------------------------------------------------#
zr1,wr1 = gausslobatto(nx1);  zs1,ws1 = gausslobatto(ny1);
zr2,wr2 = gausslobatto(nx2);  zs2,ws2 = gausslobatto(ny2);
zrd,wrd = gausslobatto(nxd);  zsd,wsd = gausslobatto(nyd);
zro     = linspace(-1,1,nxo); zso     = linspace(-1,1,nyo);

Jr1d = interpMat(zrd,zr1); Js1d = interpMat(zsd,zs1);
Jr2d = interpMat(zrd,zr2); Js2d = interpMat(zsd,zs2);
Jr21 = interpMat(zr1,zr2); Js21 = interpMat(zs1,zs2);
Jr1o = interpMat(zro,zr1); Js1o = interpMat(zso,zs1);

Dr1 = derivMat(zr1); Ds1 = derivMat(zs1);
Dr2 = derivMat(zr2); Ds2 = derivMat(zs2);
Drd = derivMat(zrd); Dsd = derivMat(zsd);

#----------------------------------------------------------------------#
# local operators -- /todo find better way
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

Iex = nothing;
Iey = nothing;
#----------------------------------------------------------------------#
# boundary conditionss
#----------------------------------------------------------------------#
ifprdcX = false
ifprdcY = false

Ix1 = Matrix(I,Ex*nx1,Ex*nx1);
Iy1 = Matrix(I,Ey*ny1,Ey*ny1);

Rx1 = Ix1[2:end-1,:];
Ry1 = Iy1[2:end-1,:];

if(ifprdcX) Rx1 = Ix1; end
if(ifprdcY) Ry1 = Iy1; end

M1 = diag(Rx1'*Rx1) * diag(Ry1'*Ry1)'; # diagonal mask

Ix1 = nothing;
Iy1 = nothing;
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
# setup
#----------------------------------------------------------------------#

# diagonal mass matrix
B1  = Jac1 .* (wx1*wy1');
B2  = Jac2 .* (wx2*wy2');
Bd  = Jacd .* (wxd*wyd');
Bi1 = 1 ./ B1;

# set up matrices for Laplace op.
G11 = @. B1 * (rx1 * rx1 + ry1 * ry1);
G12 = @. B1 * (rx1 * sx1 + ry1 * sy1);
G22 = @. B1 * (sx1 * sx1 + sy1 * sy1);

#----------------------------------------------------------------------#
# case setup
#----------------------------------------------------------------------#
# prescribe forcing, true solution, boundary condition
kx=2.
ky=3.
ut = @. sin(kx*pi*x1)*sin.(ky*pi*y1) # true solution
f  = @. ut*((kx^2+ky^2)*pi^2);       # forcing/RHS
ub = copy(ut);                       # boundary data
#----------------------------------------------------------------------#
# solve
#----------------------------------------------------------------------#

# set up RHS
b   = mass(f,B1,[],[],[]);
#b .-= lapl(ub,[],[],[],Dx1,Dy1,G11,G12,G22);
b  .= mass(b,[],M1,Qx1,Qy1);

op = LinearOperator(nx1*ny1*Ex*Ey,nx1*ny1*Ex*Ey,true,true
,v->reshape(lapl(reshape(v,nx1*Ex,ny1*Ey),M1,Qx1,Qy1,Dx1,Dy1,G11,G12,G22),nx1*ny1*Ex*Ey)
,v->reshape(lapl(reshape(v,nx1*Ex,ny1*Ey),M1,Qx1,Qy1,Dx1,Dy1,G11,G12,G22),nx1*ny1*Ex*Ey)
,nothing)

rhs = reshape(b,nx1*ny1*Ex*Ey);
u ,stats = Krylov.cg(op,rhs); stats

u = reshape(u,nx1*Ex,ny1*Ey);
#u = u + ub;

er = norm(ut-u,Inf)

