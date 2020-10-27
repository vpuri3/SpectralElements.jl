#
export mass
"""
 (v,u) = (Q*R'*v)' * B * (Q*R'*u)

 implemented as

 (QQ' * R'R * B_loc * R'R) * u_loc
"""
function mass(u,M,B,Qx,Qy);

Mu = mask(u,M);

if(length(B)==0); Bu =      Mu;
else              Bu = @. B*Mu;
end

Bu = mask(Bu,M);
Bu = gatherScatter(Bu,Qx,Qy);

return Bu
end
