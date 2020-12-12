#
#--------------------------------------#
export lapl
#--------------------------------------#
"""
 for v,u in H^1 of Omega

 (v,-del^2 u) = (vx,ux) + (vy,uy)
             := a(v,u)
              = v' * A * u
              = (Q*R'*v)'*A_l*(Q*R'*u)

 implemented as

 R'R * QQ' * A_l * u_loc

 where A_l is

 [Dr]'*[rx sx]'*[B 0]*[rx sx]*[Dr]\n
 [Ds]  [ry sy]  [0 B] [ry sy] [Ds]

"""
function lapl(u,M,QQtx,QQty,Dr,Ds,G11,G12,G22)

ur = ABu([],Dr,u);
us = ABu(Ds,[],u);

wr = @. G11*ur + G12*us;
ws = @. G12*ur + G22*us;

Au = ABu([],Dr',wr) + ABu(Ds',[],ws);

#println("in lapl, size(wr,ws,dr,ds) ",size(wr),size(ws),size(Dr),size(Ds))
#println("in lapl, size(Au1) ",size(ABu([],Dr',wr)))
#println("in lapl, size(Au2) ",size(ABu(Ds',[],ws)))
println("in lapl, size(Au ) ",size(Au))
Au = gatherScatter(Au,QQtx,QQty);
Au = mask(Au,M);

return Au
end

#--------------------------------------#
export lapl_fdm
#--------------------------------------#
"""
 Elementwise FDM Laplacian solve
"""
function lapl_fdm(b,Bi,Sx,Sy,Sxi,Syi,Di)
u = b .* Bi;
#u = ABu(Ry ,Rx ,u);
u = ABu(Syi,Sxi,u);
u = u .* Di;
u = ABu(Sy ,Sx ,u);
#u = ABu(Ry',Rx',u);
return u;
end
#--------------------------------------#
