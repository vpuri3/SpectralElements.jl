#
export mask, mask!
#
# R'*R*u - masks dirichlet boundary points
#
function mask(u,M);

    if(length(M)==0); Mu = copy(u);
    else              Mu = @. M*u;
    end
    
    return Mu
end
#
# with preallocation
#
function mask!(Mu,u,M);

    if(length(M)==0); Mu .=      u;
    else              Mu .= @. M*u;
    end
    
    return Mu
end
