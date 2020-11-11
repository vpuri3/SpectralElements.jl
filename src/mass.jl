#
#--------------------------------------#
export mass
#--------------------------------------#
"""
 (v,u) = (Q*R'*v)' * B * (Q*R'*u)

 implemented as

 (QQ' * R'R * B_loc * R'R) * u_loc
"""
function mass(u,M,B,Qx,Qy);

#u = mask(u,M);

if(length(B)==0); Bu =      u;
else              Bu = @. B*u;
end

Bu = mask(Bu,M);
Bu = gatherScatter(Bu,Qx,Qy);

return Bu
end
