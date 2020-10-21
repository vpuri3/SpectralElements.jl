#
export grad
#
# [ux]' = [rx sx]*[ur]
# [uy]  = [ry sy] [us]
#
function grad(u,Dr,Ds,rx,ry,sx,sy)

ur = ABu([],Dr,u);
us = ABu(Ds,[],u);

ux = @. rx * ur + sx * us;
uy = @. ry * ur + sy * us;

return ux,uy
end
