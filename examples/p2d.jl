#!/usr/bin/env julia

# struct simulation
#    ifvelo::Bool
#    ifconv::Bool
#    ifstks::Bool
#    iftemp::Bool
#    ngrids::Int
#end

#struct grid
#    nx1::Int
#    ny1::Int
#    Ex ::Int
#    Ey ::Int
#end

using SEM

using FastGaussQuadrature
using Plots, LinearAlgebra, SparseArrays

using LinearOperators
import Krylov

#----------------------------------------------------------------------#
# setup
#----------------------------------------------------------------------#
#ifvl = 1;    # evolve  vel field per NS eqn
#ifad = 1;    # advect  vel, sclr
#ifpr = 1;    # project vel onto a div-free subspace
#ifps = 0;    # evolve sclr per advection diffusion eqn

nx1 = 8; Ex = 5;
ny1 = 8; Ey = 5;

#nx2 = nx1-2; nxd = ceil(1.5*nx1); nxo = 10*nx1;
#ny2 = nx1-2; nyd = ceil(1.5*ny1); nyo = 10*ny1;
#----------------------------------------------------------------------#
# nodal operators
#----------------------------------------------------------------------#
zr1,wr1 =  gausslobatto(nx1); zs1,ws1 =  gausslobatto(ny1);
#zr2,wr2 =  gausslobatto(nx2); zs2,ws2 =  gausslobatto(ny2);
#zrd,wrd =  gausslobatto(nxd); zsd,wsd =  gausslobatto(nyd);
#zro     = linspace(-1,1,nxo); zso     = linspace(-1,1,nyo);

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
#wx2 = kron(ones(Ex,1),wr2); wy2 = kron(ones(Ey,1),ws2);
#wxd = kron(ones(Ex,1),wrd); wyd = kron(ones(Ey,1),wsd);

Iex = Matrix(I,Ex,Ex);
Iey = Matrix(I,Ey,Ey);

Dx1 = kron(Iex,Dr1); Dy1 = kron(Iey,Ds1);
#Dx2 = kron(Iex,Dr2); Dy2 = kron(Iey,Ds2);
#Dxd = kron(Iex,Drd); Dyd = kron(Iey,Dsd);

#Jx1d = kron(Iex,Jr1d); Jy1d = kron(Iey,Js1d);
#Jx21 = kron(Iex,Jr21); Jy21 = kron(Iey,Js21);

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

M1 = diag(Rx1'*Rx1) * diag(Ry1'*Ry1)';

Ix1 = nothing
Iy1 = nothing
Rx1 = nothing
Ry1 = nothing
#----------------------------------------------------------------------#
# mapping
#----------------------------------------------------------------------#

# Q: global -> local op, Q': local -> global
Qx1 = semq(Ex,nx1,ifperiodicX); #Qx2 = semq(Ex,nx2,ifperiodicX);
Qy1 = semq(Ey,ny1,ifperiodicY); #Qy2 = semq(Ey,ny2,ifperiodicY);

# gather scatter op
QQtx1 = Qx1*Qx1';
QQty1 = Qy1*Qy1';

#QQtx2 = Qx2*Qx2';
#QQty2 = Qy2*Qy2';

# weight for inner products
mult = ones(size(M1));
mult = ABu(QQty1,QQtx1,mult);
mult = @. 1 / mult;
#----------------------------------------------------------------------#
# geometry
#----------------------------------------------------------------------#
x1e,_ = semmesh(Ex,nx1); y1e,_ = semmesh(Ey,ny1);
#x2e,_ = semmesh(Ex,nx2); y2e,_ = semmesh(Ey,ny2);
#xde,_ = semmesh(Ex,nxd); yde,_ = semmesh(Ey,nyd);

x1,y1 = ndgrid(x1e,y1e);
#x2,y2 = ndgrid(x2e,y2e);
#xd,yd = ndgrid(xde,yde);

# deform grid with gordonhall
x1 = @. 0.5 * (x1 + 1); y1 = @. 0.5 * (y1 + 1);
#x2 = @. 0.5 * (x2 + 1); y2 = @. 0.5 * (y2 + 1);
#xd = @. 0.5 * (xd + 1); yd = @. 0.5 * (yd + 1);

Jac1,Jaci1,rx1,ry1,sx1,sy1 = jac(x1,y1,Dx1,Dy1);
#Jac2,Jaci2,rx2,ry2,sx2,sy2 = jac(x2,y2,Dx2,Dy2);
#Jacd,Jacid,rxd,ryd,sxd,syd = jac(xd,yd,Dxd,Dyd);

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
#B2  = Jac2 .* (wx2*wy2');
#Bd  = Jacd .* (wxd*wyd');
Bi1 = 1 ./ B1;

# set up matrices for Laplace op.
G11 = @. visc * B1 * (rx1 * rx1 + ry1 * ry1);
G12 = @. visc * B1 * (rx1 * sx1 + ry1 * sy1);
G22 = @. visc * B1 * (sx1 * sx1 + sy1 * sy1);
#----------------------------------------------------------------------#
# Laplace Fast Diagonalization Preconditioner
#----------------------------------------------------------------------#
rx = rx1[1];
sy = sy1[1];
Ja = Jac1[1];
Bx = Matrix(Diagonal(wr1 ./ rx));
By = Matrix(Diagonal(ws1 ./ rx));
Dx = rx*Dr1;
Dy = sy*Ds1;
Ax = Dx'*Bx*Dx;
Ay = Dy'*By*Dy;
#
#Br = Rx*Br*Rx';
#Bs = Ry*Bs*Ry';
#Ar = Rx*Ar*Rx';
#As = Ry*As*Ry';

Lx,Sx = eigen(Ax,Bx); Sxi = inv(Sx);
Ly,Sy = eigen(Ay,By); Syi = inv(Sy);
Lfdm  = Lx .+ Ly';
Lfdmi = 1 ./ Lfdm;
for j=1:ny1 for i=1:nx1
    if(abs(Lfdmi[i,j])>1e8) Lfdmi[i,j] = 0; end
end end
Sx  = kron(Iex,Sx ); Sy  = kron(Iey,Sy );
Sxi = kron(Iex,Sxi); Syi = kron(Iey,Syi);
Lfdmi = kron(ones(Ex,Ey),Lfdmi);

Li = Diagonal(reshape(Lfdmi,:));
Bi = Diagonal(reshape(Bi1  ,:));
FD = kron(Sy,Sx) * Li * kron(Syi,Sxi) * Bi;
#----------------------------------------------------------------------#
nx = nx1*Ex;
ny = ny1*Ey;
nt = nx*ny;

b =     mass(f,[],B1,[],[]);
b = b - lapl(ub,[],[],[],Dx1,Dy1,G11,G12,G22);
b =     mass(b,M1,[],Qx1,Qy1);

u = pcgLapl(b,M1,Qx1,Qy1,Dx1,Dy1,G11,G12,G22,mult)
u = u + ub;
#----------------------------------------------------------------------#
er = norm(ut-u,Inf);
print("er: ",er,"\n");
p = mesh(x1,y1,u)
display(p);
#----------------------------------------------------------------------#
nothing
