#
export gatherScatter
#
# Q*Q'*u where Q: local -> global operator
#
function gatherScatter(u,Qx,Qy);

    Qu = ABu(Qy',Qx', u); # gather
    Gu = ABu(Qy ,Qx ,Qu); # scatter

    return Gu
end
