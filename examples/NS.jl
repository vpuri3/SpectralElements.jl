using .SEM

using FastGaussQuadrature
using Plots, LinearAlgebra, SparseArrays

using LinearOperators
using Krylov

using Zygote

using DiffEqFlux,Optim,GalacticOptim,IterTools
using Flux
using Statistics
################################################################################

#----------------------------------------------------------------------#
# size
#----------------------------------------------------------------------#
nx1 = 8; Ex = 4;
ny1 = 8; Ey = 4;

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

Jr1d = interpMat(zrd,zr1); Js1d = interpMat(zsd,zs1);
Jr2d = interpMat(zrd,zr2); Js2d = interpMat(zsd,zs2);
Jr21 = interpMat(zr1,zr2); Js21 = interpMat(zs1,zs2);

Dr1 = derivMat(zr1); Ds1 = derivMat(zs1);
Dr2 = derivMat(zr2); Ds2 = derivMat(zs2);
Drd = derivMat(zrd); Dsd = derivMat(zsd);

#----------------------------------------------------------------------#
# boundary conditions
#----------------------------------------------------------------------#
ifperiodicX = true
ifperiodicY = true

Ix1 = Matrix(I,Ex*nx1,Ex*nx1);
Iy1 = Matrix(I,Ey*ny1,Ey*ny1);

Rxvx = Ix1[2:end-1,:]; Ryvx = Iy1[2:end-1,:]
Rxvy = Ix1[2:end-1,:]; Ryvy = Iy1[2:end-1,:]
Rxps = Ix1[2:end-1,:]; Ryps = Iy1[2:end-1,:]

if(ifperiodicX) Rxvx = Ix1; Rxvy = Ix1; Rxps = Ix1; end
if(ifperiodicY) Ryvx = Iy1; Ryvy = Iy1; Ryps = Iy1; end

Mvx = diag(Rxvx'*Rxvx) * diag(Ryvx'*Ryvx)'
Mvy = diag(Rxvy'*Rxvy) * diag(Ryvy'*Ryvy)'
Mps = diag(Rxps'*Rxps) * diag(Ryps'*Ryps)'

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
# x1 = @. 1. * (x1 + 0); y1 = @. 1.5/2 * (y1) + 1.5/2-.5;
# x2 = @. 1. * (x2 + 0); y2 = @. 1.5/2 * (y2) + 1.5/2-.5;
# xd = @. 1. * (xd + 0); yd = @. 1.5/2 * (yd) + 1.5/2-.5;
x1 = @. pi * (x1 + 1); y1 = @. pi * (y1 + 1);
x2 = @. pi * (x2 + 1); y2 = @. pi * (y2 + 1);
xd = @. pi * (xd + 1); yd = @. pi * (yd + 1);

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

#----------------------------------------------------------------------#
# case
#----------------------------------------------------------------------#

Re = 1
visc0 = 1/Re .+ 0 .*x1
visc1 = 1e-0 .+ 0 .*x1

T   = 1.0
CFL = 0.1

dt=1e-2
nstep = floor(T/dt)
dt = T/nstep

function kov_ex(x,y,Re)
    lam = Re/2 - sqrt(Re^2/4 + 4*pi^2);
    ue = 1 .- exp.(lam.*x).*cos.(2*pi.*y);
    ve = lam/(2*pi) * exp.(lam*x).*sin.(2*pi*y);
    return ue, ve
end
function TG(x,y,t)
    u = sin.(x).*cos.(y).*exp.(-2*visc0*t)
    v = -cos.(x).*sin.(y).*exp.(-2*visc0*t)
    return u, v
end
function usrf(xm1,ym1,xm2,ym2,time)
    Re = 1
    vxe,vye = kov_ex(xm1,ym1,Re)
    vxb = 0 .*xm1;
    vyb = 0 .*xm1;
    prb = 0 .*xm2;
    psb = 0 .*xm1;
    fvx = 0 .*xm1;
    fvy = 0 .*xm1;
    fps = 0 .*xm1;
    return vxb,vyb,prb,psb,fvx,fvy,fps
end

#----------------------------------------------------------------------#
# time
#----------------------------------------------------------------------#

global time = 0
global vx, vy = TG(x1,y1,0)#0 .*x1
# global vy = 0 .*x1
global ps = 0 .*x1
global pr = 0 .*x2

k = 3
global t_h = time.*ones(k)
global vx_h = 1 .*ones(1,1,k).*vx; global gvx_h = vx_h
global vy_h = 1 .*ones(1,1,k).*vy; global gvy_h = vy_h
global pr_h = 1 .*ones(1,1,k-1).*pr
global a,ap,b = 0,0,0

for i=1:200
    if i%10==1
        # contourf(unique(x1[1:end-1,1]),unique(y1[1,1:end-1]),SEM.ABu(Qy1',Qx1',vx.*mult1),levels=20)
        vxt,vyt = TG(x1,y1,time)
        # display(plot(heatmap(vx),heatmap(vxt),heatmap(vy),heatmap(vyt)));
        println(mean(abs,(vx.-vxt))/mean(abs,vxt))
    end

    global t_h = vcat(time,t_h[1:end-1])
    global vx_h = cat(vx,vx_h[:,:,1:end-1],dims=3)
    global vy_h = cat(vy,vy_h[:,:,1:end-1],dims=3)
    global pr_h = cat(pr,pr_h[:,:,1:end-1],dims=3)
    global time = time + dt

    global vxb,vyb,prb,psb,fvx,fvy,fps = usrf(x1,y1,x2,y2,time)

    if i<=k
        global a = reshape(SEM.diff_coeffs(t_h.-time,0),1,1,:)
        global ap = reshape(SEM.diff_coeffs(t_h[1:end-1].-time,0),1,1,:)
        global b = reshape(SEM.diff_coeffs(vcat(time,t_h).-time,1),1,1,:)
    end

    gvx = -SEM.advect(vx,vx,vy,[],[],[],Bd,Jr1d,Js1d,Dr1,Ds1,rx1,ry1,sx1,sy1)
    gvy = -SEM.advect(vy,vx,vy,[],[],[],Bd,Jr1d,Js1d,Dr1,Ds1,rx1,ry1,sx1,sy1)
    global gvx_h = cat(gvx,gvx_h[:,:,1:end-1],dims=3)
    global gvy_h = cat(gvy,gvy_h[:,:,1:end-1],dims=3)

    pr = sum(ap.*pr_h,dims=3)
    px, py = SEM.gradp(pr.+prb,[],[],B1,Jr21,Js21,Dr1,Ds1,rx1,ry1,sx1,sy1)

    bvx = sum(a.*gvx_h,dims=3)
    bvx = bvx + SEM.mass(fvx,[],Bd,Jr1d,Js1d,[],[])
    bvx = bvx - SEM.mass(sum(b[1:1,1:1,2:end].*vx_h,dims=3),[],Bd,Jr1d,Js1d,[],[])
    bvx = bvx - SEM.hlmhltz(vxb,visc0,b[1],[],Jr1d,Js1d,[],[],Bd,Dr1,Ds1,rxd,ryd,sxd,syd)
    bvx = bvx + px
    bvx = SEM.mass(bvx,Mvx,[],[],[],QQtx1,QQty1)

    bvy = sum(a.*gvy_h,dims=3)
    bvy = bvy + SEM.mass(fvy,[],Bd,Jr1d,Js1d,[],[])
    bvy = bvy - SEM.mass(sum(b[1:1,1:1,2:end].*vy_h,dims=3),[],Bd,Jr1d,Js1d,[],[])
    bvy = bvy - SEM.hlmhltz(vyb,visc0,b[1],[],Jr1d,Js1d,[],[],Bd,Dr1,Ds1,rxd,ryd,sxd,syd)
    bvy = bvy + py
    bvy = SEM.mass(bvy,Mvy,[],[],[],QQtx1,QQty1)

    opVx(v) = SEM.hlmhltz(v,visc0,b[1],Mvx,Jr1d,Js1d,QQtx1,QQty1,Bd,Dr1,Ds1,rxd,ryd,sxd,syd)
    vxhat = SEM.pcg(bvx,opVx,v->SEM.mass(v,Mvx,Bi1./b[1],[],[],QQtx1,QQty1),mult1,false)
    opVy(v) = SEM.hlmhltz(v,visc0,b[1],Mvy,Jr1d,Js1d,QQtx1,QQty1,Bd,Dr1,Ds1,rxd,ryd,sxd,syd)
    vyhat = SEM.pcg(bvy,opVy,v->SEM.mass(v,Mvy,Bi1./b[1],[],[],QQtx1,QQty1),mult1,false)

    vxhat = SEM.mask(vxhat,Mvx) + vxb
    vyhat = SEM.mask(vyhat,Mvy) + vyb

    global vx, vy, pr = SEM.pres_proj(vxhat,vyhat,pr,b[1],Mvx,Mvy,QQtx1,QQty1,QQtx2,QQty2,
                               B1,Bi1,Jr21,Js21,Dr1,Ds1,rx1,ry1,sx1,sy1,v->v,mult2)


end
