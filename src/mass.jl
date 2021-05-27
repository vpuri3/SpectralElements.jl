#
#--------------------------------------#
export mass
#--------------------------------------#
"""
 (v,u) = (Q*R'*v)' * B * (Q*R'*u)

 implemented as

 (QQ' * R'R * B_loc) * u_loc
"""
function mass(u,M,B,Jr,Js,QQtx,QQty
             ,mult);

Ju = ABu(Js,Jr,u);

if(length(B)==0); BJu =      Ju;
else              BJu = @. B*Ju;
end

Bu = ABu(Js',Jr',BJu);

#Bu = Zygote.hook(d->hmp(d),Bu);
Bu = Zygote.hook(d->d .* mult,Bu);

Bu = gatherScatter(Bu,QQtx,QQty);
Bu = mask(Bu,M);

return Bu
end
#--------------------------------------#
function mass(u,msh::Mesh)

    @unpack B = msh

    Bu = B .* u
    #Bu = gatherScatter(Bu,QQtx,QQty)
    #Bu = mask(Bu,M)

return Bu
end
#--------------------------------------#
