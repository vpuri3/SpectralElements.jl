#!/usr/bin/env julia

# struct Name
#   field::OptionalType
#   ...
# end

using SEM
using FastGaussQuadrature
using Plots, LinearAlgebra

nx1 = 16; nxd = Int(ceil(1.5*nx1)); nxo = 10*nx1;
ny1 = 16; nyd = Int(ceil(1.5*ny1)); nyo = 10*ny1;

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

# case setup
#ifvel = false;    # evolve  vel field per NS eqn
#ifadv = false;    # advect  vel, sclr
#ifpr  = true ;    # project vel onto a div-free subspace
#ift   = true ;    # evolve sclr per advection diffusion eqn
#visc  = 1;

Jac1  = ones(nx1,ny1);  # unit jacobian
Jacd  = ones(nxd,nyd);

# diagonal mass matrices
B1  = (wr1 * ws1') .* Jac1;
Bd  = (wrd * wsd') .* Jacd;
Bi1 = 1 ./ B1;
Bid = 1 ./ Bd;

# all hom. dirichlet BC
m1=[0;ones(nx1-2);0]; m2=[0;ones(ny1-2);0];
M = m1 * m2';

kx=1
ky=1
ut = sin.(kx*pi*x1).*sin.(ky*pi*y1)
f  = ut .* ((kx^2+ky^2)*pi^2);

# set up system for explicit solve
#
# -\del^2 u = f + BC
#
# solve ( Ds'*Bs*Ds \kron Br + Bs \kron Dr'*Br*Dr) * u = (Bs \kron Br) * f
# in the interior of the domain
#

Br = Diagonal(wr1);
Bs = Diagonal(ws1);
Ar = Dr1'*Br*Dr1;
As = Ds1'*Bs*Ds1;

AAr = Ar[2:end-1,2:end-1]; #remove Dirichlet BC
AAs = As[2:end-1,2:end-1];
BBr = Br[2:end-1,2:end-1];
BBs = Bs[2:end-1,2:end-1];

AA = kron(AAs,BBr) + kron(BBs,AAr);

rhs = B1 .* f;
rr  = rhs[2:end-1,2:end-1];
rr  = reshape(rr,(nx1-2)*(ny1-2),1);

# solve
uu = AA \ rr;
u  = zeros(nx1,ny1);
u[2:end-1,2:end-1] = reshape(uu,nx1-2,ny1-2);

er = norm(ut-u,Inf)
