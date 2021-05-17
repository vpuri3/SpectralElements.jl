#!/usr/bin/env julia

using SEM

using FastGaussQuadrature
using Plots, LinearAlgebra, SparseArrays

#--------------------------------------#
# setup
#--------------------------------------#
nx1 = 10; Ex = 2;
ny1 = 10; Ey = 2;
#--------------------------------------#
# nodal operators
#--------------------------------------#
zr1,wr1 =  gausslobatto(nx1)
zs1,ws1 =  gausslobatto(ny1)

Dr1 = derivMat(zr1); Ds1 = derivMat(zs1)

#----------------------------------------------------------------------#
# local operators
#----------------------------------------------------------------------#
wx1 = kron(ones(Ex,1),wr1); wy1 = kron(ones(Ey,1),ws1);

Iex = Matrix(I,Ex,Ex);
Iey = Matrix(I,Ey,Ey);

Dx1 = kron(Iex,Dr1); Dy1 = kron(Iey,Ds1);

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
Qx1 = semq(Ex,nx1,ifperiodicX)
Qy1 = semq(Ey,ny1,ifperiodicY)

# gather scatter op
QQtx1 = Qx1*Qx1';
QQty1 = Qy1*Qy1';

# mult
mult = ones(nx1*Ex,ny1*Ey);
mult = gatherScatter(mult,QQtx1,QQty1);
mult = @. 1 / mult;

mult1 = reshape(mult,:)

#----------------------------------------------------------------------#
# geometry
#----------------------------------------------------------------------#
x1e,_ = semmesh(Ex,nx1); y1e,_ = semmesh(Ey,ny1);

x1,y1 = ndgrid(x1e,y1e);

# deform grid with gordonhall
x1 = @. 0.5 * (x1 + 1); y1 = @. 0.5 * (y1 + 1);

Jac1,Jaci1,rx1,ry1,sx1,sy1 = jac(x1,y1,Dx1,Dy1);

#----------------------------------------------------------------------#
# case setup
#----------------------------------------------------------------------#
# prescribe forcing, true solution, boundary data
kx=1.
ky=1.
ut = @. sin(kx*pi*x1)*sin(ky*pi*y1) # true solution
f  = @. ut*((kx^2+ky^2)*pi^2);      # forcing/RHS
ub = copy(ut);                      # boundary data

visc = @. 1+0*x1;

#f = @. 1+0*x1;
ub= @. 0+0*x1;
#----------------------------------------------------------------------#
# setup
#----------------------------------------------------------------------#

# diagonal mass matrix
B1  = Jac1 .* (wx1*wy1');
Bi1 = 1 ./ B1;

# set up matrices for Laplace op.
G11 = @. visc * B1 * (rx1 * rx1 + ry1 * ry1);
G12 = @. visc * B1 * (rx1 * sx1 + ry1 * sy1);
G22 = @. visc * B1 * (sx1 * sx1 + sy1 * sy1);
#----------------------------------------------------------------------#
# Laplace Fast Diagonalization Preconditioner
#----------------------------------------------------------------------#
#rx = rx1[1]
#sy = sy1[1]
#Ja = Jac1[1]
#Bx = diagm(wr1 ./ rx)
#By = diagm(ws1 ./ rx)
#Dx = rx*Dr1
#Dy = sy*Ds1
#Ax = Dx'*Bx*Dx
#Ay = Dy'*By*Dy
##
##Br = Rx*Br*Rx'
##Bs = Ry*Bs*Ry'
##Ar = Rx*Ar*Rx'
##As = Ry*As*Ry'
#
#Lx,Sx = eigen(Ax,Bx); Sxi = inv(Sx);
#Ly,Sy = eigen(Ay,By); Syi = inv(Sy);
#ox=ones(size(Lx));
#oy=ones(size(Ly));
#Lfdm  = Lx*oy' + ox*Ly';
#Lfdmi = 1 ./ Lfdm;
#for j=1:ny1 for i=1:nx1
#    if(abs(Lfdmi[i,j])>1e8) Lfdmi[i,j] = 0; end
#end end
#Sx  = kron(Iex,Sx ); Sy  = kron(Iey,Sy );
#Sxi = kron(Iex,Sxi); Syi = kron(Iey,Syi);
#Lfdmi = kron(ones(Ex,Ey),Lfdmi);
#
#Li = Diagonal(reshape(Lfdmi,:));
#Bi = Diagonal(reshape(Bi1  ,:));
#FD = kron(Sy,Sx) * Li * kron(Syi,Sxi) * Bi;
#----------------------------------------------------------------------#
# Explicit system
#----------------------------------------------------------------------#
# operators
Ixl = Matrix(I,Ex*nx1,Ex*nx1);              # Identity on local vectors
Iyl = Matrix(I,Ey*ny1,Ey*ny1);
Ixg = Matrix(I,Ex*(nx1-1)+1,Ex*(nx1-1)+1);  # Identity on global vectors
Iyg = Matrix(I,Ey*(ny1-1)+1,Ey*(ny1-1)+1);
Rxl = Ixl[2:end-1,:]; Rxg = Ixg[2:end-1,:]; # restriction on loc/gl vectors
Ryl = Iyl[2:end-1,:]; Ryg = Iyg[2:end-1,:];
Rl  = kron(Ryl,Rxl);  Rg  = kron(Ryg,Rxg);
Q   = kron(Qy1,Qx1);
Dr  = kron(Iyl,Dx1);
Ds  = kron(Dy1,Ixl);
Drs = [Dr;Ds];
B   = Diagonal(reshape(B1 ,:))
g11 = Diagonal(reshape(G11,:))
g12 = Diagonal(reshape(G12,:))
g22 = Diagonal(reshape(G22,:))
G   = [g11 g12; g12 g22]
A   = Drs' * G * Drs
fl  = reshape(f,:) # forcing

# rank deficient system acting on local, unrestricted operators
M    = Rl'*Rl
QQt  = Q*Q'
Aloc = QQt * M * A # equivale to (Q * Rg' * Rg * Q') * A
Bloc = QQt * M * B
bloc = Bloc * fl
@time uloc = pcg(bloc,Aloc,x->x,mult1)
uloc = reshape(uloc,Ex*nx1,Ey*ny1)

# full rank system acting on global, restricted vectors
AA = Rg * Q' * A * Q * Rg'
BB = Rg * Q' * B * Q * Rg'
bb = Rg * Q' * B * fl;
@time uu = pcg(bb,AA,x->x)
uu = Q * Rg' * uu
uu = reshape(uu,Ex*nx1,Ey*ny1)

#----------------------------------------------------------------------#
laplOp(v) = lapl(v,M1,[],[],QQtx1,QQty1,Dx1,Dy1,G11,G12,G22,mult);
#laplFdmOp(v) = lapl_fdm(v,Bi1,Sx,Sy,Sxi,Syi,Lfdmi);

b   = mass(f,[],B1,[],[],[],[],mult)
b .-= lapl(ub,[],[],[],[],[],Dx1,Dy1,G11,G12,G22,mult)
b   = mass(b,M1,[],[],[],QQtx1,QQty1,mult)

@time u = pcg(b,laplOp,x->x,mult)
u = u + ub
#----------------------------------------------------------------------#
print("Rank deficient system, er: ",norm(ut-uloc,Inf),"\n")
print("Full rank system, er: ",norm(ut-uu,Inf),"\n")
print("Functions as operators, er: ",norm(ut-u,Inf),"\n")
p = mesh(x1,y1,uu)
display(p)
#----------------------------------------------------------------------#
nothing
