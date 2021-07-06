#
#--------------------------------------#
export gatherScatter
#--------------------------------------#
"""
Q*Q'*u where Q: local -> global operator
"""
function gatherScatter(u,QQtx::AbstractArray,QQty::AbstractArray)

# Qtu = ABu(Qy',Qx',  u); # gather
#QQtu = ABu(Qy ,Qx ,Qtu); # scatter

QQtu = ABu(QQty,QQtx,u); # gather scatter

return QQtu
end
#--------------------------------------#
function gatherScatter(u,msh::Mesh)
@unpack QQtx,QQty = msh
return gatherScatter(u,QQtx,QQty)
end
#--------------------------------------#
#Zygote.@adjoint function gatherScatter(As,Br,u)
#return ABu(As,Br,u),dv->(nothing,nothing,ABu(As',Br',dv));
#end
#--------------------------------------#
function gatherScatter(u,l2g::AbstractArray)
    
    Gu = NNlib.scatter(+,u,l2g) # gather - nnlib has its terminology reversed
    v  = Gu[l2g]
#   v  = NNlib.gather(Gu,l2g) # scatter

    return v
end
