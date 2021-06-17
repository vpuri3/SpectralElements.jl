#!/usr/bin/env julia

using Revise
using SEM
using LinearAlgebra,Plots,UnPack

#----------------------------------#
# Set up grid
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
# case setup
#----------------------------------#
# solve with Newton Raphson approach
#
# ν∇²u + G(u,x) = f(x), x∈Ω
#     u = ub,         , x∈∂Ω
#
#----------------------------------#
x1 = m1.x
y1 = m1.y

function G(u,msh::Mesh) # G(u,x⃗)
    @unpack x,y = msh
#   Gu = @. u^3
    Gu = @. -exp.(-u)
    return Gu
end
function Gdu(u,msh::Mesh) # d/du G(u,x⃗)
    @unpack x,y = msh
#   Gdu = @. 3u^2
    Gdu = @. exp.(-u)
    return Gdu
end

ν  = @. 1+0*x1
u  = @. 0+0*x1 # initial guess (must agree with boundary data)
δu = @. 0+0*x1 # initialize
f  = @. 1+0*x1

# contrived example
kx = 3
ky = 3
ut = @. 0 + sin(kx*pi*x1)*sin(ky*pi*y1) # true solution
f  = @. ut*((kx^2+ky^2)*pi^2)         # forcing/RHS
f += G(ut,m1)

# boundary condition
# 'N': Neumann (homogeneous)
# 'D': Dirichlet (inhomogeniety applied through boundary data)
M1 = generateMask(['D','D','D','D'],m1) # [xmin,xmax,ymin,ymax]

#----------------------------------#
# solve linearized homogeneous equation for δu
#----------------------------------#
function residual(v,msh::Mesh)
    resi  = mass(f,msh)
    resi -= mass(G(v,msh),msh) .+ ν.*lapl(v,msh)
    resi .= mask(resi,M1)
    resi .= gatherScatter(resi,m1)
    return resi
end

function opHlmz(v,k,msh::Mesh) # LHS op
    Hu = hlmz(v,ν,k,msh)
    Hu = gatherScatter(Hu,msh)
    Hu = mask(Hu,M1)
    return Hu
end

for i=1:100
    local g = G(u,m1)
    local k = Gdu(u,m1)

    # LHS
    opLHS(v) = opHlmz(v,k,m1)

    # RHS
    b  = mass(f,m1)                  # forcing
    b -= ν.*lapl(u,m1) .+ mass(g,m1) # from LHS
    b .= mask(b,M1)
    b .= gatherScatter(b,m1)

    # solve
    pcg!(δu,b,opLHS,mult=m1.mult,ifv=false)
    u .+= δu

    r  = residual(u,m1);rr = norm(r,Inf);println("iter $i, res norm $rr")
    if(rr < 1e-8) break end
end
#----------------------------------#
er=ut-u; er=norm(er,Inf); println("Error ∞ norm: $er")
plt = meshplt(u,m1)
display(plt)
#----------------------------------#
nothing
