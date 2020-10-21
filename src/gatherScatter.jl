#
export gatherScatter
#
# Q*Q'*u where Q: local -> global operator
#
function gatherScatter(u,Qx,Qy);

 Qtu = ABu(Qy',Qx',  u); # gather
QQtu = ABu(Qy ,Qx ,Qtu); # scatter

return QQtu
end
