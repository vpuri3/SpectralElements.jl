#
export ABu
#
# v <- kron(As,Br) * u
#
function ABu(As,Br,u)

    if(length(As)==0 && length(Br)==0); v=u;     return v;
    elseif(length(As)==0);              v=Br*u ; return v;
    elseif(length(Br)==0);              v=u*As'; return v;
    end
    
    v = Br*u*As';

    return v;
end
