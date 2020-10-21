#
export lapl
#
# (v,-\del^2 u) = (vx,ux) + (vy,uy)
#
# implemented as
#
# QQ' * R'R *
#
#  [Dr]'*[rx sx]'*[B 0 ]*[rx sx]*[Dr]*u
#  [Ds]  [ry sy]  [0 B ] [ry sy] [Ds]    
#
# * R'R * u_loc
#
function lapl(u,M,Qx,Qy,Dr,Ds,G11,G12,G22)

Mu = mask(u,M);

ur = ABu([],Dr,Mu);
us = ABu(Ds,[],Mu);

wr = @. G11*ur + G12*us;
ws = @. G12*ur + G22*us;

Au = ABu([],Dr',wr) + ABu(Ds',[],ws);

Au = mask(Au,M);
Au = gatherScatter(Au,Qx,Qy);

return Au
end
