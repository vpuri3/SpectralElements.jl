#!/usr/bin/env julia

using SpectralElements

using Plots, LinearAlgebra, SparseArrays

#--------------------------------------#
# setup
#--------------------------------------#
nx1 = 10; Ex = 2;
ny1 = 10; Ey = 2;
#--------------------------------------#
# nodal operators
#--------------------------------------#
zr1,wr1 = SpectralElements.FastGaussQuadrature.gausslobatto(nx1)
zs1,ws1 = SpectralElements.FastGaussQuadrature.gausslobatto(ny1)

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
@time uloc = pcg(bloc,Aloc;mult=mult1)
uloc = reshape(uloc,Ex*nx1,Ey*ny1)

#----------------------------------------------------------------------#
laplOp(v) = lapl(v,M1,[],[],QQtx1,QQty1,Dx1,Dy1,G11,G12,G22,mult);
#laplFdmOp(v) = lapl_fdm(v,Bi1,Sx,Sy,Sxi,Syi,Lfdmi);

b   = mass(f,[],B1,[],[],[],[],mult)
b .-= lapl(ub,[],[],[],[],[],Dx1,Dy1,G11,G12,G22,mult)
b   = mass(b,M1,[],[],[],QQtx1,QQty1,mult)

@time u = pcg(b,laplOp;mult=mult)
u = u + ub
#----------------------------------------------------------------------#
print("Rank deficient system, er: ",norm(ut-uloc,Inf),"\n")
print("Functions as operators, er: ",norm(ut-u,Inf),"\n")
p = meshplt(x1,y1,uu)
display(p)
#----------------------------------------------------------------------#
return
