#
#--------------------------------------#
export lapl
#--------------------------------------#
"""
 for v,u in H¹₀(Ω)

 (v,-∇² u) = (vx,ux) + (vy,uy)\n
          := a(v,u)\n
           = v' * A * u\n
           = (Q*R'*v)'*A_l*(Q*R'*u)\n
           = v'*R*Q'*A_l*Q*R'*u\n

 implemented as

 R'R * QQ' * A_l * u_loc
 where A_l is

 [Dr]'*[rx sx]'*[B 0]*[rx sx]*[Dr]
 [Ds]  [ry sy]  [0 B] [ry sy] [Ds]

 = [Dr]' * [G11 G12]' * [Dr]
   [Ds]    [G12 G22]    [Ds]

"""
function lapl(u::AbstractArray
             ,msh::Mesh)

    @unpack Dr,Ds,G11,G12,G22 = msh

    Au = laplace(u,Dr,Ds,G11,G12,G22)
#   Au = gatherScatter(Au,msh)
#   Au = mask(Au,M)

    return Au
end
#--------------------------------------#
function lapl(u::AbstractArray
             ,ν::AbstractArray
             ,msh::Mesh)

    # -∇⋅(ν∇u)

    return ν .* lapl(u,msh)
end
#--------------------------------------#
function lapl(u::AbstractArray
             ,msh1::Mesh
             ,msh2::Mesh)
    # Dealiased implementation
    return lapl(u,msh1)
end
#--------------------------------------#
function lapl(u,M,Jr,Js,QQtx,QQty,Dr,Ds
             ,G11,G12,G22,mult)

Au = u;

Au=laplace(Au,Jr,Js,Dr,Ds,G11,G12,G22);

#Au = Zygote.hook(d->hmp(d),Au);
Au = Zygote.hook(d->d .* mult,Au);

Au = gatherScatter(Au,QQtx,QQty);
Au = mask(Au,M);

return Au
end
#--------------------------------------#
function laplace(u,Dr,Ds,G11,G12,G22)

ur = ABu([],Dr,u)
us = ABu(Ds,[],u)

wr = @. G11*ur + G12*us
ws = @. G12*ur + G22*us

Au = ABu([],Dr',wr) + ABu(Ds',[],ws)

return Au
end
#--------------------------------------#
function laplace(u,Jr,Js,Dr,Ds
                ,G11,G12,G22)

# dealiased

ur = ABu([],Dr,u)
us = ABu(Ds,[],u)

Jur = ABu(Js,Jr,ur)
Jus = ABu(Js,Jr,us)

vr = @. G11*Jur + G12*Jus
vs = @. G12*Jur + G22*Jus

wr = ABu(Js',Jr',vr)
ws = ABu(Js',Jr',vs)

Au = ABu([],Dr',wr) + ABu(Ds',[],ws)

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
