#
#--------------------------------------#
export ABu
#--------------------------------------#
"""
 (As ⊗ Br) * u
"""
function ABu(As::Matrix,Br::Matrix,u::Array)

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
#--------------------------------------#
Zygote.@adjoint function ABu(As,Br,u)
return ABu(As,Br,u),dv->(nothing,nothing,ABu(As',Br',dv));
end
#--------------------------------------#
export ABu1
#--------------------------------------#
"""
 (As ⊗ Br) * u
"""
function ABu1(As,Br,u)

    m = size(As)[1]
    n = size(Br)[1]
    E = size(u )[3]

    v = zeros(m,n,E)

    ABu1!(v,As,Br,u,E=E)

return v
end
#--------------------------------------#
export ABu1!
#--------------------------------------#
"""
 v = (As ⊗ Br) * u

 To adopt this (general) implementation of the tensor product operation,
 we need to get the following to work with arrays of shape (nx,ny,ne)

 1. geometry: x,y <= semreshape
 2. mask: <= construct in square domain, and then do sem_reshape
 3. gather scatter ops <= global ⟺   local numbering and do sem_reshape;

 sem_reshape -- should only be done when generating Mesh object

 focus on deformed([-1,1]²) for now. think about general geomtries later

"""
function ABu1!(v,As,Br,u;E=5)

    @views for ie=1:E
        v[:,:,ie] .= Br * u[:,:,ie] * As'
    end

return
end
#--------------------------------------#
