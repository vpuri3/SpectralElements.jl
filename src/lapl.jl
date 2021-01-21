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

 = [Dr]'*[G11 G12]'*[Dr]\n
   [Ds]  [G12 G22]  [Ds]
"""
function lapl(u,M,Jr,Js,QQtx,QQty,Dr,Ds
             ,G11,G12,G22,mult)

Au = u;

Au = laplace(Au,Jr,Js,Dr,Ds,G11,G12,G22);

Zygote.ignore() do
Au = mass(Au,M,[],[],[],QQtx,QQty);
end

#Au = Zygote.hook(d->hmp(d),Au);

return Au
end
#--------------------------------------#
function laplace(u,Jr,Js,Dr,Ds
                ,G11,G12,G22)

ur = ABu([],Dr,u);
us = ABu(Ds,[],u);

Jur = ABu(Js,Jr,ur);
Jus = ABu(Js,Jr,us);

vr = @. G11*Jur + G12*Jus;
vs = @. G12*Jur + G22*Jus;

wr = ABu(Js',Jr',vr);
ws = ABu(Js',Jr',vs);

Au = ABu([],Dr',wr) + ABu(Ds',[],ws);

return Au
end
#--------------------------------------#
#export lapl_fdm
#--------------------------------------#
#"""
# Elementwise FDM Laplacian solve
#"""
#function lapl_fdm(b,Bi,Sx,Sy,Sxi,Syi,Di)
#u = b .* Bi;
##u = ABu(Ry ,Rx ,u);
#u = ABu(Syi,Sxi,u);
#u = u .* Di;
#u = ABu(Sy ,Sx ,u);
##u = ABu(Ry',Rx',u);
#return u;
#end
#--------------------------------------#
