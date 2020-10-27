#
export mask
"""
 Mu = (R'*R)*u
 masks dirichlet boundary points
"""
function mask(u,M);

    if(length(M)==0); Mu = copy(u);
    else              Mu = @. M*u;
    end
    
    return Mu
end
