#
export mass
#
# (v,u) = (R'*Q*v)' * B * (R'*Q*u)
#
function mass(u,B,M,Qx,Qy);

    Mu = mask(u,M);
    
    if(length(B)==0); Bu=   Mu;
    else              Bu=B.*Mu;
    end
    
    Bu = gatherScatter(Bu,Qx,Qy);
    Bu = mask(Bu,M);

    return Bu
end
