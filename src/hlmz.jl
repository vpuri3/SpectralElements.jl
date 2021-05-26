#
#--------------------------------------#
export hlmz
#--------------------------------------#
"""
 for v,u in H^1 of Omega

 (v,(-visc*del^2 + k) u )
"""
function hlmz(u,k,M
             ,Bd,Jr,Js,QQtx,QQty,Dr,Ds
             ,G11,G12,G22,mult)
   
Au =   lapl(u,[],Jr,Js,[],[],Dr,Ds,G11,G12,G22,[])
ku = k*mass(u,[],Bd,Jr,Js,[],[],[])

Hu = Au + ku

Hu = mass(Hu,[],Bd,Jr1d,Js1d,QQtx,QQty,mult)

return Hu
end
#--------------------------------------#
#
