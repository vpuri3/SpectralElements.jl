#!/usr/bin/env julia

using SEM, OrdinaryDiffEq, Flux

using Plots, LinearAlgebra, SparseArrays

#--------------------------------------#
# setup
#--------------------------------------#
nx1 = 8; Ex = 2;
ny1 = 8; Ey = 2;
#--------------------------------------#
# nodal operators
#--------------------------------------#
zr1,wr1 = SEM.FastGaussQuadrature.gausslobatto(nx1)
zs1,ws1 = SEM.FastGaussQuadrature.gausslobatto(ny1)

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
#x1 = @. 0.5 * (x1 + 1); y1 = @. 0.5 * (y1 + 1);

Jac1,Jaci1,rx1,ry1,sx1,sy1 = jac(x1,y1,Dx1,Dy1);

#----------------------------------------------------------------------#
# case setup
#----------------------------------------------------------------------#
kx=1.0
ky=1.0
kt=1.0
ν =1.0

utrue(x,y,t) = sin(kx*pi*x)*sin.(ky*pi*y)*cos(kt*pi*t)
setIC(x,y)   = utrue(x,y,0.0)

function setForcing(x,y,t)
    ut = utrue(x,y,t)
    f  = ut*((kx^2+ky^2)*pi^2*ν)
    f -= sin(kx*pi*x)*sin(ky*pi*y)*sin(kt*pi*t)*(kt*pi)
end

# callback
function cond(u,t,integrator)
    istep = integrator.iter
    cond = (istep % 1) == 0
    return cond
end

function affect!(integrator)
    istep = integrator.iter
    u = integrator.u
    p = integrator.p
    t = integrator.t
    dt = integrator.dt

    ut = utrue.(xx,yy,t)
    er = norm(ut-u,Inf)
    println("Step=$istep, Time=$t, dt=$dt, er=$er")

    ul  = reshape(Q*Rg'*u ,nx1*Ex,ny1*Ey)
#   dul = reshape(Q*Rg'*du,nx1*Ex,ny1*Ey)
    plt = meshplt(x1,y1,ul)
    plt = plot!(zlims=(-1,1))
    display(plt)
    return
end

cb = DiscreteCallback(cond,affect!,save_positions=(false,false))

#----------------------------------------------------------------------#
# setup
#----------------------------------------------------------------------#

# diagonal mass matrix
B1  = Jac1 .* (wx1*wy1');
Bi1 = 1 ./ B1;

# set up matrices for Laplace op.
G11 = @. ν * B1 * (rx1 * rx1 + ry1 * ry1);
G12 = @. ν * B1 * (rx1 * sx1 + ry1 * sy1);
G22 = @. ν * B1 * (sx1 * sx1 + sy1 * sy1);
#----------------------------------------------------------------------#
# Linear System
#----------------------------------------------------------------------#

Ixl = Matrix(I,Ex*nx1,Ex*nx1);              # Identity on local vectors
Iyl = Matrix(I,Ey*ny1,Ey*ny1);
Ixg = Matrix(I,Ex*(nx1-1)+1,Ex*(nx1-1)+1);  # Identity on global vectors
Iyg = Matrix(I,Ey*(ny1-1)+1,Ey*(ny1-1)+1);
Rxl = Ixl[2:end-1,:]; Rxg = Ixg[2:end-1,:]; # restriction on loc/gl vectors
Ryl = Iyl[2:end-1,:]; Ryg = Iyg[2:end-1,:];
Rl  = kron(Ryl,Rxl)
Rg  = kron(Ryg,Rxg)
Q   = kron(Qy1,Qx1)
Dr  = kron(Iyl,Dx1);
Ds  = kron(Dy1,Ixl);
Drs = [Dr;Ds];
B   = Diagonal(reshape(B1 ,:))
g11 = Diagonal(reshape(G11,:))
g12 = Diagonal(reshape(G12,:))
g22 = Diagonal(reshape(G22,:))
G   = [g11 g12; g12 g22]
A   = Drs' * G * Drs

# local vectors
xl  = reshape(x1,:)
yl  = reshape(y1,:)

# global
xx = Rg * Q' * xl
yy = Rg * Q' * yl

#----------------------------------------------------------------------#
# full rank global system
AA = Rg * Q' * A * Q * Rg'
BB = Rg * Q' * B * Q * Rg'
Bb = Rg * Q' * B

Big = Rg * Q' * reshape(Bi1,:)
BBi = Diagonal(Big)
precondB = BBi \ LinearAlgebra.I

#----------------------------------------------------------------------#
"""
 mass matrix form
"""

function Mdudt!(Mdu,u,p,t)
    f = setForcing.(xl,yl,t)
    Mdu = -AA*u + Bb*f
    return Mdu
end

u0   = setIC.(xx,yy)
dt   = 0.01
tspn = (0.0,1.0)

func = ODEFunction(Mdudt!;mass_matrix=BB)
prob = ODEProblem(func,u0,tspn)
sol  = solve(prob,
#            Rodas5();
#            Rodas5(linsolve=LinSolveGMRES(Pl=precondB));
             Rodas5(linsolve=LinSolveGMRES());
             saveat=0.1,
             callback=cb,
            )

@show sol.retcode
err = sol.u - [utrue.(xx,yy,sol.t[i]) for i=axes(sol.t,1)]
ee  = maximum.(abs.(err[i]) for i=axes(sol.t,1))
#----------------------------------------------------------------------#
"""
 standard form - mass-matrix inversion inside ODEFunciton
"""

function dudt!(du,u,p,t)
    f = setForcing.(xl,yl,t)
    rhs = -AA*u + Bb*f
    pcg!(du,rhs,BB;opM=x->x.*Big,ifv=false)
    return du
end

u0   = setIC.(xx,yy)
dt   = 0.01
tspn = (0.0,1.0)

func = ODEFunction(dudt!)
prob = ODEProblem(func,u0,tspn)
sol  = solve(prob, Rodas5(); saveat=0.1, callback=cb)

@show sol.retcode
err = sol.u - [utrue.(xx,yy,sol.t[i]) for i=axes(sol.t,1)]
ee  = maximum.(abs.(err[i]) for i=axes(sol.t,1))
#----------------------------------------------------------------------#
return ee
