using .SEM

using FastGaussQuadrature
using Plots, LinearAlgebra, SparseArrays

using LinearOperators
using Krylov

using  Zygote

using DiffEqFlux,Optim
using Flux
using Statistics

function get_disc(nx1,ny1,Ex,Ey)
#----------------------------------------------------------------------#
# size
#----------------------------------------------------------------------#
# nx1 = 2; Ex = 16;
# ny1 = 2; Ey = 16;

nxd = 8;
nyd = 8;

nxp = 2;
nyp = 2;

#----------------------------------------------------------------------#
# nodal operators
#----------------------------------------------------------------------#
zr1,wr1 = gausslobatto(nx1);  zs1,ws1 = gausslobatto(ny1);
zrd,wrd = gausslobatto(nxd);  zsd,wsd = gausslobatto(nyd);
zrp     = linspace(-1,1,nxp); zsp     = linspace(-1,1,nyp);

Jr1d = interpMat(zrd,zr1); Js1d = interpMat(zsd,zs1);
Jr1p = interpMat(zrp,zr1); Js1p = interpMat(zsp,zs1);

Dr1 = derivMat(zr1); Ds1 = derivMat(zs1);
Drd = derivMat(zrd); Dsd = derivMat(zsd);

#----------------------------------------------------------------------#
# mapping
#----------------------------------------------------------------------#

ifperiodicX = false
ifperiodicY = false

# Q: global -> local op, Q': local -> global
Qx1 = semq(Ex,nx1,ifperiodicX);
Qy1 = semq(Ey,ny1,ifperiodicY);

# gather scatter op
QQtx1 = Qx1*Qx1';
QQty1 = Qy1*Qy1';

# weight for inner products
mult1 = ones(nx1*Ex,ny1*Ey);
mult1 = SEM.gatherScatter(mult1,QQtx1,QQty1);
mult1 = @. 1 / mult1;

#----------------------------------------------------------------------#
# geometry
#----------------------------------------------------------------------#
x1e,_ = semmesh(Ex,nx1); y1e,_ = semmesh(Ey,ny1);
xde,_ = semmesh(Ex,nxd); yde,_ = semmesh(Ey,nyd);

x1,y1 = ndgrid(x1e,y1e);
xd,yd = ndgrid(xde,yde);

# deform grid with gordonhall
x1 = @. 0.5 * (x1 + 1); y1 = @. 0.5 * (y1 + 1);
xd = @. 0.5 * (xd + 1); yd = @. 0.5 * (yd + 1);

Jac1,Jaci1,rx1,ry1,sx1,sy1 = jac(x1,y1,Dr1,Ds1);
Jacd,Jacid,rxd,ryd,sxd,syd = jac(xd,yd,Drd,Dsd);

wx1 = kron(ones(Ex,1),wr1); wy1 = kron(ones(Ey,1),ws1);
wxd = kron(ones(Ex,1),wrd); wyd = kron(ones(Ey,1),wsd);

# diagonal mass matrix
B1  = Jac1 .* (wx1*wy1');
Bd  = Jacd .* (wxd*wyd');
Bi1 = 1 ./ B1;

return Jr1d, Js1d, Jr1p, Js1p, Dr1, Ds1, Drd, Dsd, QQtx1, QQty1, mult1,
       x1, y1, rxd, ryd, sxd, syd, B1, Bd
end

#----------------------------------------------------------------------#
# functions
#----------------------------------------------------------------------#

#----------------------------------------------------------------------#
# things too add
Ex = 16
Ey = 16

nx1 = 2
ny1 = 2
nxd = 8
nyd = 8
#----------------------------------------------------------------------#

Ax1 = Matrix(1.0I, nx1, nx1); Ax1[end,1:end-1] .= -1
Ay1 = Matrix(1.0I, ny1, ny1); Ay1[end,1:end-1] .= -1
Axd = Matrix(1.0I, nxd, nxd); Axd[end,1:end-1] .= -1
Ayd = Matrix(1.0I, nyd, nyd); Ayd[end,1:end-1] .= -1
Ad = Matrix(1.0I, nxd*nyd, nxd*nyd); Ad[end,1:end-1] .= -1

li = Chain(Conv((3,3),5=>16,pad=1,stride=1,swish),
           Conv((3,3),16=>16,pad=1,stride=1,swish),
           Conv((3,3),16=>16,pad=1,stride=1,swish),
           MaxPool((2,2)),
           Conv((3,3),16=>32,pad=1,stride=1,swish),
           Conv((3,3),32=>64,pad=1,stride=1,swish),
           Conv((3,3),64=>(nx1-1)*nxd+(ny1-1)*nyd,pad=1,stride=1))

p0,re = Flux.destructure(li)

function Jlearn(p,x,y,visc,f,M)
    infields = cat(x,y,visc,f,M,dims=3)
    infields = reshape(infields, (size(infields)...,1))
    Jmats = re(p)(infields)

    Jmats = permutedims(Jmats, [3,1,2,4])

    Jr = reshape([cat(reshape(Jmats[1:(nx1-1)*nxd,(i-1)%Ex+1,Base.ceil(Int,i/Ex)],nxd,nx1-1),
                 zeros(nxd,1),dims=2)' for i=1:Ex*Ey], Ex, Ey)
    Js = reshape([cat(reshape(Jmats[(nx1-1)*nxd+1:(nx1-1)*nxd+(ny1-1)*nyd,(i-1)%Ex+1,Base.ceil(Int,i/Ex)],nyd,ny1-1),
                 zeros(nyd,1),dims=2)' for i=1:Ex*Ey], Ex, Ey)

    Jr = broadcast(transpose, Ref(Ax1).*Jr)
    Js = broadcast(transpose, Ref(Ay1).*Js)

    # B = reshape([reshape(Ad*cat(reshape(Jmats[(nx1-1)*nxd+(ny1-1)*nyd+1:end,
    #              (i-1)%Ex+1,Base.ceil(Int,i/Ex)],nxd*nyd-1,1),0,dims=1),nxd,nyd) for i=1:Ex*Ey], Ex, Ey)
    # B = vcat([hcat(B[i,:]...) for i=1:size(B,1)]...)

    return Jr, Js
end


function problem(p)

    Jr, Js = Jlearn(p,x1,y1,visc1,f1,M1)
    Jr, Js, B = Jr.+Ref(Jr1d), Js.+Ref(Js1d), Bd

    visc  = visc1
    viscd = SEM.ABu(Js,Jr,visc);
    G11 = @. viscd * B * (rxd * rxd + ryd * ryd);
    G12 = @. viscd * B * (rxd * sxd + ryd * syd);
    G22 = @. viscd * B * (sxd * sxd + syd * syd);

    lhs(v) = SEM.lapl(v,M1,Jr,Js,QQtx1,QQty1,Dr1,Ds1,G11,G12,G22,mult1);

    rhs = SEM.mass(f1,M1,B,Jr,Js,QQtx1,QQty1);

    return lhs, rhs
end

function problemt(x)

    Jr, Js, B = Jr1d, Js1d, Bd

    visc  = visc1
    viscd = SEM.ABu(Js,Jr,visc);
    G11 = @. viscd * B * (rxd * rxd + ryd * ryd);
    G12 = @. viscd * B * (rxd * sxd + ryd * syd);
    G22 = @. viscd * B * (sxd * sxd + syd * syd);

    lhs(v) = SEM.lapl(v,M1,Jr,Js,QQtx1,QQty1,Dr1,Ds1,G11,G12,G22,mult1);

    rhs = SEM.mass(f1,M1,B,Jr,Js,QQtx1,QQty1);

    return lhs, rhs
end

function opM(v) # preconditioner
    return v
end

function solver(opA,rhs,adj::Bool)
    if adj
        rhs = mass(rhs,M1,mult1,[],[],QQtx1,QQty1);
        out = pcg(rhs,opA,opM,mult1,false)
        return out;
    else
        return pcg(rhs,opA,opM,mult1,false);
    end
end

function model(p)
    u = SEM.linsolve(p,problem,solver,mult1,M1,QQtx1,QQty1) # adjoint support thru Zygote
end

function modelt()
    u = SEM.linsolve(0,problemt,solver,mult1,M1,QQtx1,QQty1) # adjoint support thru Zygote
end

function loss(p,ut)
    u = model(p)
    u = ABu(Js1p,Jr1p,u)
    l = sum(abs2,u.-ut)
    return l, u
end

function zloss(p)
   Jr, Js = Jlearn(p,x1,y1,visc1,f1,M1)
   Jr = vcat([hcat(Jr[i,:]...) for i=1:size(Jr,1)]...)
   Js = vcat([hcat(Js[i,:]...) for i=1:size(Js,1)]...)
   l = sum(abs2,Jr)+sum(abs2,Js)
   return l, (Jr,Js)
end

global param = p0
callback = function (p, l, pred; doplot = true)
  println(l)
  global param = p
  return false
end

#----------------------------------------------------------------------#
# setup
#----------------------------------------------------------------------#

function case_setup(f,x1,y1,xlbd,xrbd,ylbd,yrbd)
    f1 = f.(x1,y1)
    visc1 = @. 1 + 0*x1
    M1 = ones(size(x1))
    if xlbd; M1[1,:] .= 0; end
    if xrbd; M1[end,:] .= 0; end
    if ylbd; M1[:,1] .= 0; end
    if xrbd; M1[:,end] .= 0; end
    return M1, visc1, f1
end

force(x,y) = @. 5*sin(2*x*2*pi)-10*cos(5*y*2*pi)
BCs = (true,false,true,false)

# Spectral true
Jr1d, Js1d, Jr1p, Js1p, Dr1, Ds1, Drd, Dsd, QQtx1, QQty1, mult1,
x1, y1, rxd, ryd, sxd, syd, B1, Bd = get_disc(8,8,16,16)
M1, visc1, f1 = case_setup(force,x1,y1,BCs...)

ut = ABu(Js1p,Jr1p,modelt())

# FEM
Jr1d, Js1d, Jr1p, Js1p, Dr1, Ds1, Drd, Dsd, QQtx1, QQty1, mult1,
x1, y1, rxd, ryd, sxd, syd, B1, Bd = get_disc(2,2,16,16)
M1, visc1, f1 = case_setup(force,x1,y1,BCs...)

baseline = ABu(Js1p,Jr1p,modelt())

bloss = sum(abs2,baseline.-ut)

#----------------------------------------------------------------------#
# training
#----------------------------------------------------------------------#
iftrain = true
if iftrain
# pretrain IC to 0
zres = DiffEqFlux.sciml_train(p->zloss(p),p0,ADAM(.005),cb=callback,maxiters=500)

opt = ADAM(1e-5)
# opt.eta = 1e-6
result = DiffEqFlux.sciml_train(p->loss(p,ut),zres.minimizer,opt,cb=callback,maxiters=100)
#result = DiffEqFlux.sciml_train(p->loss(p,ut),loss,zres.minimizer,opt,
#                    Iterators.repeated((ut,),100),cb=callback)
best_res = result.minimizer
end
#----------------------------------------------------------------------#
# test
#----------------------------------------------------------------------#

force(x,y) = @. 10*sin(2*x*2*pi)+5*cos(5*y*2*pi)

# Spectral true
Jr1d, Js1d, Jr1p, Js1p, Dr1, Ds1, Drd, Dsd, QQtx1, QQty1, mult1,
x1, y1, rxd, ryd, sxd, syd, B1, Bd = get_disc(8,8,16,16)
M1, visc1, f1 = case_setup(force,x1,y1,BCs...)

@time utt = ABu(Js1p,Jr1p,modelt())

# FEM
Jr1d, Js1d, Jr1p, Js1p, Dr1, Ds1, Drd, Dsd, QQtx1, QQty1, mult1,
x1, y1, rxd, ryd, sxd, syd, B1, Bd = get_disc(2,2,16,16)
M1, visc1, f1 = case_setup(force,x1,y1,BCs...)

baselinet = ABu(Js1p,Jr1p,modelt())
@time predt = ABu(Js1p,Jr1p,model(best_res))

blosst = sum(abs2,baselinet.-utt)
predlosst = sum(abs2,predt.-utt)

println(predlosst < blosst)
