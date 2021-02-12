#
#--------------------------------------#
export ABu
export ABub
#--------------------------------------#
"""
 (As kron Br) * u
"""
function ABu(As,Br,u)

if typeof(As)<:AbstractArray{<:AbstractArray{}}

    m,n = size(u)
    mb,nb = size(Br[1])
    ma,na = size(As[1])
    Ex = Int(m/nb); Ey = Int(n/na)

    if length(Br)==0
        ABu = u
    else
        u = reshape([u[nb*((i-1)%Ex)+1:nb*((i-1)%Ex+1),
                       na*(Base.ceil(Int,i/Ex)-1)+1:na*Base.ceil(Int,i/Ex)] for i=1:Ex*Ey], Ex, Ey)
        As = broadcast(transpose, As);
        ABu = @. Br*u*As
        ABu = vcat([hcat(ABu[i,:]...) for i=1:size(ABu,1)]...)
    end

else

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
        tmp   = Zygote.Buffer(u,m,n)#zeros(m,n);
        for i=1:Ey
            ii=Int((i-1)*ma+1):Int(i*ma);
            jj=Int((i-1)*na+1):Int(i*na);
            tmp[:,ii] = ABu[:,jj]*As'
        end
        ABu = copy(tmp)
    end
end
# ABu = Zygote.hook(dABu->display(heatmap(dABu)),ABu)
return ABu
end

#--------------------------------------#
# Zygote.@adjoint function ABu(As,Br,u)
# return ABu(As,Br,u),dv->(nothing,nothing,ABu(As',Br',dv));
# end
#--------------------------------------#
# """
# (As kron Br) * u
# """
# function ABu(As,Br,u)
# m,n = size(u);
# u = u[:]
# if(length(As)==0 && length(Br)==0); v = u;
# elseif(length(As)==0);              v = Br*u ;
# elseif(length(Br)==0);              v = u*As';
# else                                v = Br*u*As';
# end
# v = reshape(v,m,n)
# v = Zygote.hook(dABu->display(heatmap(dABu)),v)
# return v;
# end
#--------------------------------------#
#export ABu3
#--------------------------------------#
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
#--------------------------------------#
