#
export pcg
#
# Conjugate gradient for Laplace solve
#
function pcg(u,M,Qx,Qy,Dr,Ds,G11,G12,G22)

Au = lapl(u,M,Qx,Qy,Dr,Ds,G11,G12,G22)

return x
end
