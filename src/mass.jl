#
#--------------------------------------#
export mass
#--------------------------------------#
"""
 (v,u) = (Q*R'*v)' * B * (Q*R'*u)

 implemented as

 (QQ' * R'R * B_loc) * u_loc
"""
function mass(u,M,B,Jr,Js,QQtx,QQty);

Ju = ABu(Js,Jr,u);

if(length(B)==0); BJu =      Ju;
else              BJu = @. B*Ju;
end

Bu = ABu(tpose(Js),tpose(Jr),BJu);

Bu = mask(Bu,M);
Bu = gatherScatter(Bu,QQtx,QQty);

return Bu
end
