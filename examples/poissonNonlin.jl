#!/usr/bin/env julia

using Revise
using SEM
using LinearAlgebra,Plots,BenchmarkTools

#----------------------------------#
Ex = 8; nr1 = 8;
Ey = 8; ns1 = 8;

function deform(x,y)
    x = @. 0.5*(x+1)
    y = @. 0.5*(y+1)
    return x,y
end

ifperiodic = [false,false] # [xdir, ydir]

m1 = Mesh(nr1,ns1,Ex,Ey,deform,ifperiodic)

#----------------------------------#
# solve with Newton Raphson approximation
#
# ∇²u + G(u,x) = f(x)
#----------------------------------#
x1 = m1.x
y1 = m1.y

G(v,msh::Mesh) = @. v^2
Gdu(v,mesh::Mesh) = @. 2v

ν  = @. 1+0*x1

kx = 2.0
ky = 2.0
ut = @. sin(kx*pi*x1)*sin(ky*pi*y1) # true solution
f  = @. ut*((kx^2+ky^2)*pi^2)       # forcing/RHS
f += G(ut,m1)

ub  = @. 0+0*x1 # boundary data
δub = @. 0+0*x1

M1 = generateMask(['D','D','D','D'],m1) # [xmin,xmax,ymin,ymax]

function residual(v,msh::Mesh)
    resi  = mass(f,msh)
    resi -= mass(G(v,msh),msh) .+ ν.*lapl(v,msh)
    return resi
end

function opHlmz(v,k,msh::Mesh) # LHS op
    Hu = hlmz(v,ν,k,msh)
    Hu = gatherScatter(Hu,msh)
    Hu = mask(Hu,M1)
    return Hu
end

u  = @. 0+0*x1 # initial guess
δu = @. 0+0*x1
for i=1:10
    local g = G(u,m1)
    local k = Gdu(u,m1)

    # LHS
    opLHS(v) = opHlmz(v,k,m1)

    # RHS
    b  = mass(f,m1)
    b -= ν.*lapl(u,m1) .+ mass(g,m1)
    b  = mask(b,M1)
    b  = gatherScatter(b,m1)

    # solve
    pcg!(δu,b,opLHS,mult=m1.mult,ifv=false)
    u .+= δu

    r  = residual(u,m1) # residual calculation is not adding up...
    r .= mask(r,M1)
    r .= gatherScatter(r,m1)

    bb = norm(b ,Inf); println("At iter $i, RHS norm is $bb")
    uu = norm(δu,Inf); println("At iter $i, δu  norm is $uu")
    rr = norm(r ,Inf); println("At iter $i, res norm is $rr")

    if(rr < 1e-6) break end
end
#----------------------------------#
plt = meshplt(u,m1)
display(plt)
#----------------------------------#
nothing
