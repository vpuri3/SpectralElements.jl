#!/usr/bin/env julia

using SEM
using LinearAlgebra,Plots

#----------------------------------#
Ex = 8; nr1 = 8; #nr2 = nr1-2; nrd = Int(ceil(1.5*nr1));
Ey = 8; ns1 = 8; #ns2 = ns1-2; nsd = Int(ceil(1.5*ns1));

function deform(x,y)
    x,y = SEM.annulus(0.5,1.0,2pi,x,y)
#   x = @. 0.5*(x+1)
#   y = @. 0.5*(y+1)
    return x,y
end

ifperiodic = [false, true]

m1 = Mesh(nr1,ns1,Ex,Ey,deform,ifperiodic)
#m2 = Mesh(nr2,ns2,Ex,Ey,deform,ifperiodic)
#md = Mesh(nrd,nsd,Ex,Ey,deform,ifperiodic)

#----------------------------------#
# case setup
#----------------------------------#

x1 = m1.x
y1 = m1.y

# prescribe forcing, true solution, boundary data
kx=2.0
ky=2.0
ut = @. sin(kx*pi*x1)*sin.(ky*pi*y1) # true solution
f  = @. ut*((kx^2+ky^2)*pi^2);       # forcing/RHS
ub = copy(ut);                       # boundary data

visc = @. 1+0*x1

f = @. 1+0*x1
ub= @. 0+0*x1

M1 = generateMask(['D','D','N','N'],m1)
Mb = generateMask(['N','N','N','N'],m1)

#----------------------------------#
# ops for PCG
#----------------------------------#

function opLapl(v)
    Au = lapl(v,m1)
    Au = gatherScatter(Au,m1.QQtx,m1.QQty)
    Au = mask(Au,M1)
    return Au
end
#
function opFDM(v)
    return v
end
#----------------------------------#
b =     mass(f ,m1)
b = b - lapl(ub,m1)

b = mask(b ,M1)
b = gatherScatter(b,m1)

@time u = pcg(b,opLapl,opM=opFDM,mult=m1.mult,ifv=true)
u = u + ub
#----------------------------------#
er = norm(ut-u,Inf)
print("er: ",er,"\n")
p = meshplt(u,m1)
display(p)
#----------------------------------#
nothing
