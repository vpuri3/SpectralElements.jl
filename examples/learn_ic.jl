using .SEM

using FastGaussQuadrature
using Plots, LinearAlgebra, SparseArrays

using LinearOperators
using Krylov

using  Zygote

using DiffEqFlux,Optim,GalacticOptim,IterTools
using Flux
using Statistics
################################################################################

# Define mesh
Ex, Ey = 16, 16
nx1, ny1 = 2, 2
nxd, nyd = 4, 4
lgrid = (nx1,ny1,Ex,Ey,nxd,nxd)
hgrid = (8,8,Ex,Ey,12,12)

################################################################################

function get_disc(nx1,ny1,Ex,Ey,nxd,nyd)
#----------------------------------------------------------------------#
# size
#----------------------------------------------------------------------#

nxp = 2;
nyp = 2;

#----------------------------------------------------------------------#
# nodal operators
#----------------------------------------------------------------------#
zr1,wr1 = gausslobatto(nx1);  zs1,ws1 = gausslobatto(ny1);
zrd,wrd = gausslobatto(nxd);  zsd,wsd = gausslobatto(nyd);
zrp = range(-1,stop=1,length=nxp); zsp = range(-1,stop=1,length=nyp)

Jr1d = interpMat(zrd,zr1); Js1d = interpMat(zsd,zs1);
Jr1p = interpMat(zrp,zr1); Js1p = interpMat(zsp,zs1);

Dr1 = derivMat(zr1); Ds1 = derivMat(zs1);
Drd = derivMat(zrd); Dsd = derivMat(zsd);

#----------------------------------------------------------------------#
# mapping
#----------------------------------------------------------------------#

ifperiodicX, ifperiodicY = false, false
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

return x1, y1, Jr1d, Js1d, Jr1p, Js1p, QQtx1, QQty1, Dr1, Ds1, Bd, rxd, ryd, sxd, syd, mult1

# return Jr1d, Js1d, Jr1p, Js1p, Dr1, Ds1, Drd, Dsd, QQtx1, QQty1, mult1,
#        x1, y1, rxd, ryd, sxd, syd, B1, Bd
end

# visc,f,M,x,y,Jr,Js,QQx,QQy,Dr,Ds,B,rx,ry,sx,sy,mult
#----------------------------------------------------------------------#
# functions
#----------------------------------------------------------------------#

Ax1 = Matrix(1.0I, nx1, nx1); Ax1[end,1:end-1] .= -1
Ay1 = Matrix(1.0I, ny1, ny1); Ay1[end,1:end-1] .= -1
Axd = Matrix(1.0I, nxd, nxd); Axd[end,1:end-1] .= -1
Ayd = Matrix(1.0I, nyd, nyd); Ayd[end,1:end-1] .= -1
Ad = Matrix(1.0I, nxd*nyd, nxd*nyd); Ad[end,1:end-1] .= -1

li = Chain(Conv((3,3),5=>32,pad=1,stride=1,swish),
           Conv((3,3),32=>32,pad=1,stride=1,swish),
           #MaxPool((2,2)),
           Conv((2,2),32=>128,pad=0,stride=2,swish),
           Conv((3,3),128=>128,pad=1,stride=1,swish),
           Conv((3,3),128=>(nx1-1)*nxd+(ny1-1)*nyd,pad=1,stride=1))

p0,re = Flux.destructure(li)

function Jlearn(p,M,visc,f,x,y)

    infields = cat(x,y,visc,f,M,dims=3)
    infields = reshape(infields, (size(infields)...,1))
    Jmats = re(p)(infields)

    Jmats = permutedims(Jmats, [3,1,2,4])

    Jr = reshape([cat(reshape(Jmats[1:(nx1-1)*nxd,(i-1)%Ex+1,ceil(Int,i/Ex)],nxd,nx1-1),
                 zeros(nxd,1),dims=2)' for i=1:Ex*Ey], Ex, Ey)
    Js = reshape([cat(reshape(Jmats[(nx1-1)*nxd+1:(nx1-1)*nxd+(ny1-1)*nyd,(i-1)%Ex+1,ceil(Int,i/Ex)],nyd,ny1-1),
                 zeros(nyd,1),dims=2)' for i=1:Ex*Ey], Ex, Ey)

    Jr = broadcast(transpose, Ref(Ax1).*Jr)
    Js = broadcast(transpose, Ref(Ay1).*Js)

    # Jr = reshape([reshape(Jmats[1:(nx1-0)*nxd,(i-1)%Ex+1,ceil(Int,i/Ex)],nxd,nx1-0) for i=1:Ex*Ey], Ex, Ey)
    # Js = reshape([reshape(Jmats[(nx1-0)*nxd+1:(nx1-0)*nxd+(ny1-0)*nyd,
    #                      (i-1)%Ex+1,ceil(Int,i/Ex)],nyd,ny1-0) for i=1:Ex*Ey], Ex, Ey)

    # B = reshape([reshape(Ad*cat(reshape(Jmats[(nx1-1)*nxd+(ny1-1)*nyd+1:end,
    #              (i-1)%Ex+1,ceil(Int,i/Ex)],nxd*nyd-1,1),0,dims=1),nxd,nyd) for i=1:Ex*Ey], Ex, Ey)
    # B = vcat([hcat(B[i,:]...) for i=1:size(B,1)]...)

    return Jr, Js
end


function problem(p,input)

    M,visc,f,x,y,Jr,Js,_,_,QQx,QQy,Dr,Ds,B,rx,ry,sx,sy,mult = input

    Jrc, Jsc = Jlearn(p,M,visc,f,x,y)
    Jrc, Jsc = Jrc.+Ref(Jr), Jsc.+Ref(Js)

    viscd = SEM.ABu(Js,Jr,visc);
    G11 = @. viscd * B * (rx * rx + ry * ry)
    G12 = @. viscd * B * (rx * sx + ry * sy)
    G22 = @. viscd * B * (sx * sx + sy * sy)

    lhs(v) = SEM.lapl(v,M,Jrc,Jsc,QQx,QQy,Dr,Ds,G11,G12,G22,mult)

    rhs = SEM.mass(f,M,B,Jr,Js,QQx,QQy)

    return lhs, rhs
end

function problemt(null,input)

    M,visc,f,x,y,Jr,Js,_,_,QQx,QQy,Dr,Ds,B,rx,ry,sx,sy,mult = input

    viscd = SEM.ABu(Js,Jr,visc)
    G11 = @. viscd * B * (rx * rx + ry * ry)
    G12 = @. viscd * B * (rx * sx + ry * sy)
    G22 = @. viscd * B * (sx * sx + sy * sy)

    lhs(v) = SEM.lapl(v,M,Jr,Js,QQx,QQy,Dr,Ds,G11,G12,G22,mult)

    rhs = SEM.mass(f,M,B,Jr,Js,QQx,QQy)

    return lhs, rhs
end

function opM(v) # preconditioner
    return v
end

function solver(opA,rhs,adj::Bool,mult,M,QQx,QQy)
    if adj
        rhs = mass(rhs,M,mult,[],[],QQx,QQy);
        out = pcg(rhs,opA,opM,mult,false)
        return out;
    else
        return pcg(rhs,opA,opM,mult,false);
    end
end

function model(p,input)
    M,visc,f,x,y,Jr,Js,Jro,Jso,QQx,QQy,_,_,_,_,_,_,_,mult = input
    u = SEM.linsolve(p,input,problem,solver,mult,M,QQx,QQy) # adjoint support thru Zygote
    # Jrc, Jsc = Jlearn(p,M,visc,f,x,y)
    # Jrc, Jsc = Jrc.+Ref(Jr), Jsc.+Ref(Js)
    u = ABu(Jso,Jro,u)
end

function modelt(input)
    M,_,_,_,_,Jr,Js,Jro,Jso,QQx,QQy,_,_,_,_,_,_,_,mult = input
    u = SEM.linsolve(0,input,problemt,solver,mult,M,QQx,QQy) # adjoint support thru Zygote
    u = ABu(Jso,Jro,u)
end

function loss(p,input,ut)
    u = model(p,input)
    uth, utl = ut
    bl = sum(abs2,utl.-uth)
    l = sum(abs2,u.-uth)
    return (l-bl), u
end

function zloss(p,input)
    infields = input[1:5]
    Jr, Js = Jlearn(p,infields...)
    Jr = vcat([hcat(Jr[i,:]...) for i=1:size(Jr,1)]...)
    Js = vcat([hcat(Js[i,:]...) for i=1:size(Js,1)]...)
    l = sum(abs2,Jr)+sum(abs2,Js)
    return l, (Jr,Js)
end

global param = p0
callback = function (p, l, pred; doplot = true)
  global param = p
  test_loss = zeros(5)
  train_loss = zeros(5)
  for (i,x) in enumerate(dataxyt)
      input,utf = x
      u = model(p,input)
      uth, utl = utf
      l = sum(abs2,u.-uth)
      test_loss[i] = l
  end
  for (i,x) in enumerate(dataxy)
      input,utf = x
      u = model(p,input)
      uth, utl = utf
      l = sum(abs2,u.-uth)
      train_loss[i] = l
  end
  println(mean(test_loss./databt)-1," ",mean(train_loss./databt)-1)
  return false
end
callback2 = function (p, l, pred)
    println(l)
    return false
end

#----------------------------------------------------------------------#
# setup
#----------------------------------------------------------------------#

function case_setup(f,x1,y1,xlbd,xrbd,ylbd,yrbd)
    f1 = f(x1,y1)
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

discMatsl = get_disc(lgrid...)
casel = case_setup(force,discMatsl[1],discMatsl[2],BCs...)
inputl = (casel...,discMatsl...)

discMatsh = get_disc(hgrid...)
caseh = case_setup(force,discMatsh[1],discMatsh[2],BCs...)
inputh = (caseh...,discMatsh...)

baseline = modelt(inputl)
ut = modelt(inputh)

bloss = sum(abs2,baseline.-ut)

#----------------------------------------------------------------------#
# training
#----------------------------------------------------------------------#

function gen_train(n)
    datax = []; dataxy = []; datab = []; datac = []
    i = 1
    while i<=n
        BCs = (true,false,true,false)
        rs = rand(4).-.5
        force(x,y) =    (rs[1]*10)*sin.((rs[2]*8)*x*pi) +
                        (rs[3]*10)*cos.((rs[4]*8)*y*pi)

        discMatsl = get_disc(lgrid...)
        casel = case_setup(force,discMatsl[1],discMatsl[2],BCs...)
        inputl = (casel...,discMatsl...)

        discMatsh = get_disc(hgrid...)
        caseh = case_setup(force,discMatsh[1],discMatsh[2],BCs...)
        inputh = (caseh...,discMatsh...)

        uth = modelt(inputh)
        utl = modelt(inputl)
        bloss = sum(abs2,utl.-uth)

        if !(bloss > 10 || isnan(bloss))
            push!(datax,(inputl,))
            push!(dataxy,(inputl,(uth,utl)))
            push!(datab,bloss)
            push!(datac,caseh)
            i=i+1
        end
    end
    return datax, dataxy, datab, datac
end

datax, dataxy, datab, datac = gen_train(5)
dataxt, dataxyt, databt, datact = gen_train(5)

opt = ADAM(1e-4)

println("Pretraining...")
# pretrain IC to 0
optfun = OptimizationFunction((θ,p,x)->zloss(θ,x), GalacticOptim.AutoZygote())
optprob = OptimizationProblem(optfun, p0)
zres = GalacticOptim.solve(optprob, ADAM(0.001), ncycle(datax,10),
                           cb=callback2, maxiters=1e3)

println("Training...")
optfun = OptimizationFunction((θ,p,x,y)->loss(θ,x,y), GalacticOptim.AutoZygote())
optprob = OptimizationProblem(optfun, zres.minimizer)
result = GalacticOptim.solve(optprob, opt, ncycle(dataxy,10), cb=callback, maxiters=1e3)

best_res = result.minimizer

pred = model(best_res,inputl)

#----------------------------------------------------------------------#
# test
#----------------------------------------------------------------------#

force(x,y) = @. 5*sin(3*x*2*pi)+2*cos(2*y*2*pi)
BCs = (true,false,true,false)

discMatsl = get_disc(lgrid...)
casel = case_setup(force,discMatsl[1],discMatsl[2],BCs...)
inputl = (casel...,discMatsl...)

discMatsh = get_disc(hgrid...)
caseh = case_setup(force,discMatsh[1],discMatsh[2],BCs...)
inputh = (caseh...,discMatsh...)

baseline = modelt(inputl)

@time utt = modelt(inputh)
baselinet = modelt(inputl)
@time predt = model(best_res,inputl)

blosst = sum(abs2,baselinet.-utt)
predlosst = sum(abs2,predt.-utt)

println("")
println("Baseline test loss: ", blosst)
println("")
println("")
println("Test loss: ", predlosst)
println("")

println(predlosst < blosst)

nxpl = 5*nx1
nypl = 5*ny1
zrpl = range(-1,stop=1,length=nxpl); zspl = range(-1,stop=1,length=nypl)
zrd,wrd = gausslobatto(nxd);  zsd,wsd = gausslobatto(nyd)
Jrdpl = interpMat(zrpl,zrd); Jsdpl = interpMat(zspl,zsd)
tpl(u) = ABu(Jsdpl,Jrdpl,u)

test_loss = zeros(5)
for (i,x) in enumerate(dataxyt)
    input,utf = x
    u = model(best_res,input)
    uth, utl = utf
    l = sum(abs2,u.-uth)
    test_loss[i] = l
end

println(mean(test_loss.-databt))
