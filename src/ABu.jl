#
export ABu, ABu!
#
# v <- kron(As,Br) * u
#
function ABu(As,Br,u)

#   v = Array{Float64}(undef,size(Br,1),size(As,1));

if(length(As)==0 && length(Br)==0); v = copy(u);
elseif(length(As)==0);              v = Br*u ;
elseif(length(Br)==0);              v = u*As';
else                                v = Br*u*As';
end

return v;
end
#
#   with prealloaction
#
#function ABu!(v,As,Br,u)
#
#    if(length(As)==0 && length(Br)==0); v .= u;
#    elseif(length(As)==0);              v .= Br*u ;
#    elseif(length(Br)==0);              v .= u*As';
#    else                                v .= Br*u*As';
#    end
#
#    return
#end
