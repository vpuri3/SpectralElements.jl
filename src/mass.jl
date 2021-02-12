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

if typeof(Jr)<:AbstractArray{<:AbstractArray{}}
    Jrt = broadcast(transpose,Jr)
    Jst = broadcast(transpose,Js)
else
    Jrt = Jr'; Jst = Js'
end
Bu = ABu(Jst,Jrt,BJu);

Bu = mask(Bu,M);
Bu = gatherScatter(Bu,QQtx,QQty);

return Bu
end
