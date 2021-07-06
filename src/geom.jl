#
#--------------------------------------#
export gordonHall
#--------------------------------------#
"""
 Transfinite interpolation
"""
function gordonHall(xrm,xrp,xsm,xsp,yrm,yrp,ysm,ysp,zr,zs)

ze  = [-1,1]
Jer = interpMat(zr,ze)
Jes = interpMat(zs,ze)

xv = [xrm[1]   xrp[1]
      xrm[end] xrp[end]]

yv = [yrm[1]   yrp[1]
      yrm[end] yrp[end]]

xv = ABu(Jes,Jer,xv)
yv = ABu(Jes,Jer,yv)

#display(mesh(xv,yv,0*xv,0,90))

x = ABu([],Jer,vcat(xrm',xrp')) .+ ABu(Jes,[],hcat(xsm,xsp)) .- xv
y = ABu([],Jer,vcat(yrm',yrp')) .+ ABu(Jes,[],hcat(ysm,ysp)) .- yv

#display(mesh(x,y,0*x,0,90))

return x,y
end
#--------------------------------------#
export annulus
#--------------------------------------#
"""
 Convert (r,s) in [-1,1]^2 grid
 internal/external radii r0,r1,
 and angular extent span
"""
function annulus(r,s;r0=0.5,r1=1.0,span=2pi)

R  = @. (r1-r0)/2*(r+1) + r0
th = @. span   /2*(s+1) + 0.

x = @. R * cos(th)
y = @. R * sin(th)

return x,y
end
#--------------------------------------#
#"""
# Transforms [-1,1]^2 to annulus with
# internal/external radii r0,r1,
# and angular extent span
#"""
#function annulus(r0,r1,span,zr,zs)
#
#ze  = [-1.,1.]
#Jer = interpMat(zr,ze);
#Jes = interpMat(zs,ze);
#
#xrp = 0*zs;
#yrp = Jes*[-r0,-r1];
#
#spn = span-pi/2;
#
#xrm = Jes*[r0;r1]*cos(spn);
#yrm = Jes*[r0;r1]*sin(spn);
#
#as = Jer*[-pi/2,spn];
#xsm = r0 .* cos.(as);
#ysm = r0 .* sin.(as);
#
#xsp = r1 .* cos.(as);
#ysp = r1 .* sin.(as);
#
#pl=plot(xsm,ysm ,label="sm");
#pl=plot!(xsp,ysp,label="sp");
#pl=plot!(xrm,yrm,label="rm");
#pl=plot!(xrp,yrp,label="rp");
#pl=xlabel!("x")
#pl=ylabel!("y")
#display(pl)
#
#return gordonHall(xrm,xrp,xsm,xsp,yrm,yrp,ysm,ysp,zr,zs)
#end
#--------------------------------------#
