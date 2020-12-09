using GaussQuadrature, LinearAlgebra
using SparseArrays
using Plots
using Zygote

# Mesher
function hex2d(start, finish, dims)
    x = range(start[1],finish[1],length=dims[1]+1)
    y = range(start[2],finish[2],length=dims[2]+1)
    # pts = hcat([[x[i],y[j]] for i=1:length(x) for j=1:length(y)]...)'
    plist = Iterators.product(x,y)

    elist = []
    # plist = [Tuple(pts[i,:]) for i=1:size(pts,1)]
    pts = [getindex.(plist,1)[:] getindex.(plist,2)[:]]
    for i = 1:dims[1]
        for j = 1:dims[2]
            e = [(x[i],y[j]),(x[i+1],y[j]),(x[i+1],y[j+1]),(x[i],y[j+1])]
            push!(elist,e)
        end
    end

    # p = unique(vcat(elist...))
    t = hcat([findfirst.(eachrow(elist[i].==reshape(collect(plist),1,:))) for i=1:length(elist)]...)'
    # p = hcat([[p[i]...] for i=1:length(p)]...)'
    return x, y, pts, t
end

# Bilinear quad shape functions
c = [-1 -1
      1 -1
      1  1
     -1  1]

p(c) = hcat(ones(size(c,1),1),
            c[:,1], c[:,2],
            c[:,1].*c[:,2])
dp1(c) = hcat(zeros(size(c,1),1),
              0*c[:,1].+1, 0*c[:,2],
              c[:,2])
dp2(c) = hcat(zeros(size(c,1),1),
              0*c[:,1], 0*c[:,2].+1,
              c[:,1])

α = p(c)\I(size(c,1))
N(c) = p(c)*α
Dr = dp1(c)*α
Ds = dp2(c)*α

# Quadrature
lgN = 4
gp,gw = legendre(lgN)
rs = Iterators.product(gp,gp)
rs = [getindex.(rs,1)[:] getindex.(rs,2)[:]]
# rs = hcat([[gp[i],gp[j]] for i=1:lgN for j=1:lgN]...)'

BM = (gw*gw')[:]

Jgp = N(rs)

# Element Poisson
function Ke(pts,κ,t,e)
    xyr = Jgp*Dr*pts[t[e,:],:]
    xys = Jgp*Ds*pts[t[e,:],:]
    jac = xyr[:,1].*xys[:,2] .- xys[:,1].*xyr[:,2]

    rx = xys[:,2]./jac; ry = -xys[:,1]./jac
    sx = -xyr[:,2]./jac; sy = xyr[:,1]./jac

    Rx = [[rx[i] sx[i];ry[i] sy[i]] for i=1:lgN^2]
    JB = [[jac[i].*BM[i] 0;0 jac[i].*BM[i]] for i=1:lgN^2]

    κe = [reshape(reshape(κ[:,:,t[e,:]],4,:)*Jgp',2,2,:)[:,:,i] for i=1:lgN^2]

    G = cat(broadcast((rx,jb,κ)->rx'*jb*κ*rx,Rx,JB,κe)...,dims=3)
    G = vcat([diagm(G[1,1,:]) diagm(G[1,2,:])],
             [diagm(G[2,1,:]) diagm(G[2,2,:])])

    D = [Jgp*Dr;Jgp*Ds]
    Ke = D'*G*D
end

function be(pts,f,t,e)
    xyr = Jgp*Dr*pts[t[e,:],:]
    xys = Jgp*Ds*pts[t[e,:],:]
    jac = xyr[:,1].*xys[:,2] .- xys[:,1].*xyr[:,2]
    JB = diagm(jac.*BM)

    temp = Jgp'*JB
    be = temp*Jgp*f[t[e,:]]
end

function integrate(u,pts,t,e)
    xyr = Jgp*Dr*pts[t[e,:],:]
    xys = Jgp*Ds*pts[t[e,:],:]
    jac = xyr[:,1].*xys[:,2] .- xys[:,1].*xyr[:,2]
    ue = Jgp*u
    return sum(ue.*BM.*jac)
end

function grad(u,pts,t,e)
    xyr = Dr*pts[t[e,:],:]
    xys = Ds*pts[t[e,:],:]
    jac = xyr[:,1].*xys[:,2] .- xys[:,1].*xyr[:,2]

    rx = xys[:,2]./jac; ry = -xys[:,1]./jac
    sx = -xyr[:,2]./jac; sy = xyr[:,1]./jac

    Rx = vcat([diagm(rx) diagm(sx)],
              [diagm(ry) diagm(sy)])

    return reshape(Rx*[Dr;Ds]*u,:,2)
end

Zygote.@adjoint diagm(x::AbstractVector) = diagm(x), dy -> (diag(dy),)
Zygote.@adjoint sparse(i,j,v,m,n) = sparse(i,j,v,m,n), dy ->
                        (nothing,nothing,dy[CartesianIndex.(i,j)],nothing,nothing)
Zygote.@adjoint sparsevec(i,v,n) = sparsevec(i,v,n), dy -> (nothing,dy[i],nothing)

function setup(κ,f,M,bb)
    # Assembly
    K = spzeros(nn,nn)
    b = zeros(nn)
    # K = []
    # b = []

    for e = 1:numel
        idxs = Zygote.ignore() do
            idxs = Iterators.product(t[e,:],t[e,:])
            idxs = [getindex.(idxs,1)[:] getindex.(idxs,2)[:]]
        end
        K = K+sparse(idxs[:,1],idxs[:,2],Ke(pts,κ,t,e)[:],nn,nn)
        b = Array(b+sparsevec(t[e,:],be(pts,f,t,e),nn))
    end

    Mi = Int.(.!Bool.(M))
    lhs = K.*M.+I(nn).*Mi
    rhs = b.*M.+bb

    return lhs, rhs
end

function solver(lhs, rhs, adj::Bool)
    if !adj
        return lhs\rhs#Krylov.cg(op,rhs)[1]
    else
        return lhs'\rhs#Krylov.cg(op',rhs)[1]
    end
end

#----------------------------------------------------------------------#
# set up case
#----------------------------------------------------------------------#

# Mesh
start = (0,0); finish = (1,1); dims = (30,30)
x,y,pts,t = hex2d(start,finish,dims)
numel = size(t,1)
nn = size(pts,1)
toplot(u) = reshape(u,dims[1]+1,dims[2]+1)'

V = 0.4
pow = 5
ε = 1e-3
α = 0#1e-8
f = 0*ones(nn).+1e-2
k(a) = @. ε + (1-ε)*a^pow

# BCs
xlbi = findall(pts[:,1].==start[1])
xrbi = findall(pts[:,1].==finish[1])
ylbi = findall(pts[:,2].==start[2])
yrbi = findall(pts[:,2].==finish[2])

bis = [xlbi;ylbi]
Tbs = [(xlbi*0);
       (ylbi*0)]
# bis = [xlbi;xrbi;ylbi;yrbi]
# Tbs = [(xlbi*0);(xrbi*0);(ylbi*0);(yrbi*0)]

M = ones(nn); M[bis] .= 0
bb = zeros(nn); bb[bis] .= Tbs

function problem(p)
    κ = reshape(k(p),1,1,:).*I(2)
    return setup(κ,f,M,bb)
end

function model(p)
    lhs, rhs = problem(p)
    lhs\rhs
end

af(p) = @. 0.5*(tanh(p)+1)
function loss(p)
    a = af(p)
    u = model(a)

    ft = f.*u
    l = 0
    for e=1:numel
        l += integrate(ft[t[e,:]],pts,t,e)
        l += α*integrate(sum(grad(a[t[e,:]],pts,t,e).^2,dims=2),pts,t,e)
    end

    # l = 0.5*mean(abs2,u.-ut)
    return l, u
end

p0 = ones(nn)

global param = p0
callback = function (p, l, pred)
  display(l)
  global param = p
  return false
end

result = DiffEqFlux.sciml_train(loss, result.minimizer, ADAM(.001),
                                cb = callback, maxiters = 500)
heatmap(x,y,toplot(af(param)),linewidth=0.,clim=(0.5,1))

# gradient(p->loss(p)[1],p0)
