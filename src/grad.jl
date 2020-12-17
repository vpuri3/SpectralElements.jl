#
#--------------------------------------#
export grad
#--------------------------------------#
"""
 [Dx]*u = [rx sx]*[Dr] * u\n
 [Dy]   = [ry sy] [Ds]
"""
function grad(u,Dr,Ds,rx,ry,sx,sy)

ur = ABu([],Dr,u);
us = ABu(Ds,[],u);

ux = @. rx * ur + sx * us;
uy = @. ry * ur + sy * us;

return ux,uy
end
