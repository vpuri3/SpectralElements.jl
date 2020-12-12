#
#export ABu
#"""
# (As kron Br) * u
#"""
#function ABu(As,Br,u)
#
#    if(length(As)==0 && length(Br)==0); v = u;
#    elseif(length(As)==0);              v = Br*u ;
#    elseif(length(Br)==0);              v = u*As';
#    else                                v = Br*u*As';
#    end
#
#return v;
#end

#----------------------------------------------------------------------#
export ABu
#----------------------------------------------------------------------#
"""
 (As kron Br) * u
"""
function ABu(As,Br,u)

    m,n = size(u);

    Bu = u;
    if(length(Br)!=0)
        mb,nb = size(Br);
        m     = Int(m*mb/nb);
        Bu    = reshape(Bu,nb,:)
        Bu    = Br * Bu;
        Bu    = reshape(Bu,m,n);
    end

    ABu = Bu;
    if(length(As)!=0)
        ma,na = size(As);
        Ey    = n/na;
        n     = Int(Ey*ma);
        tmp   = zeros(m,n);
        for i=1:Ey
            ii=Int((i-1)*ma+1):Int(i*ma);
            jj=Int((i-1)*na+1):Int(i*na);
            tmp[:,ii] = ABu[:,jj]*As';
        end
        ABu = tmp
    end

return ABu
end
#----------------------------------------------------------------------#
#Zygote.@adjoint function ABu(As,Br,u)
#    return ABu(As',Br',u);
#end
#----------------------------------------------------------------------#
#export ABu3
#----------------------------------------------------------------------#
#"""
# (As kron Br) * u
#"""
#function ABu3(As,Br,u)
#
#    m,n,E = size(u);
#
#    Bu = u;
#    if(length(Br)!=0)
#        m ,nb = size(Br);
#        Bu    = reshape(Bu,nb,:);
#        Bu    = Br * Bu;
#    end
#
#    ABu = Bu;
#    if(length(As)!=0)
#        n ,na = size(As);
#        ABu   = reshape(ABu,:,na)
#        ABu   = ABu * As';
#    end
#
#    ABu = reshape(ABu,m,n,E);
#
#return ABu
#end
#----------------------------------------------------------------------#
