#
#--------------------------------------#
export DSS
#--------------------------------------#
"""
Q*Q'*u where Q: local -> global operator
"""
function DSS(u,l2g,g2l)
    
    Qu   = NNlib.scatter(+,u,l2g)
    QQtu = NNlib.gather(Qu,g2l)

    return v
end
