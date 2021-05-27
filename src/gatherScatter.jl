#
#--------------------------------------#
export gatherScatter
#--------------------------------------#
"""
Q*Q'*u where Q: local -> global operator
"""
function gatherScatter(u,QQtx,QQty);

# Qtu = ABu(Qy',Qx',  u); # gather
#QQtu = ABu(Qy ,Qx ,Qtu); # scatter

QQtu = ABu(QQty,QQtx,u); # gather scatter

return QQtu
end
#--------------------------------------#
#Zygote.@adjoint function gatherScatter(As,Br,u)
#return ABu(As,Br,u),dv->(nothing,nothing,ABu(As',Br',dv));
#end
#--------------------------------------#
