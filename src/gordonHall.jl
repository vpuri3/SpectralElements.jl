#
export gordonHall
"""
 Transfinite interpolation
"""
function gordonHall(xrm,xrp,xsm,xsp,yrm,yrp,ysm,ysp,zr,zs)
#
ze  = [-1;1];
Jer = interpMat(zr,ze);
Jes = interpMat(zs,ze);

xv = [xrm[[1 end]]';xrp[[1;end]]'];
yv = [yrm[[1 end]]';yrp[[1;end]]'];
xv = ABu(Jes,Jer,xv);
yv = ABu(Jes,Jer,yv);

x = ABu([],Jer,[xrm';xrp']) .+ ABu(Jes,[],[xsm xsp]) .- xv;
y = ABu([],Jer,[yrm';yrp']) .+ ABu(Jes,[],[ysm ysp]) .- yv;

mesh(x,y,0*x)

return x,y
end
