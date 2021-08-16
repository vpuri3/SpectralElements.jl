#
using SEM, Plots
import FastGaussQuadrature

nr = 8
np = 100

zr,_ = FastGaussQuadrature.gausslobatto(nr)
zp   = linspace(-1,1,np)

J = interpMat(zp,zr)

## basis

plt = plot()
plt = plot!(title="Gauss Lobbato Legendre Polynomials, ψᵢ(x)")
#plt = plot!(xlims=(-1,1))

fr = @. 2 + sin(pi*zr)

for i=1:nr
    global plt = plot!(zp,J[:,i]*fr[i],width=2,label=:none)
end

plt = plot!(zp,J*fr,width=2,color=:black,label=:none)
savefig(plt,"basis.png")

## elem

#plt = plot()
#plt = plot!(xlims=(-1,3))
#
#for i=1:nr
#    global plt = plot!(0 .+zp,J[:,i],width=2,label=:none)
#end
#savefig(plt,"elem1.png")
#
#plt = plot()
#plt = plot!(xlims=(-1,3))
#
#for i=1:nr
#    global plt = plot!(2 .+zp,J[:,i],width=2,label=:none)
#end
#savefig(plt,"elem2.png")

## indicator func
#zp = linspace(-1,3,1000)
#id1 = -1 .<= zp .<= 1
#id2 =  1 .<= zp .<= 3
#plt = plot()
#plt = plot!(xlims=(-1,3),ylims=(-0.2,1.2))
#plt = plot!(zp,id1,width=2,label=:none,color=:black)
#savefig(plt,"id1.png")
#plt = plot()
#plt = plot!(xlims=(-1,3),ylims=(-0.2,1.2))
#plt = plot!(zp,id2,width=2,label=:none,color=:black)
#savefig(plt,"id2.png")


