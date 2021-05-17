#
#--------------------------------------#
export linearFEM
#--------------------------------------#
"""
 1D linear FEM operators in (0,1)
"""
function linearFEM(N)

	n  = N + 1
	dz = 1/N
    z  = linspace(0,1,n)

    dn = ones(n)
    dN = ones(N)

    A = Tridiagonal(-1*dN,2*dn ,-1*dN) / dz # == D'*B*D
    B = Tridiagonal(1dN/6,2dn/3,1dN/6) * dz
    C = Tridiagonal(-dN/2,0dn  ,dN/2)        # == B * D
    D = Tridiagonal( 0dN ,-1dn ,1dN)   / dz  # attempt second order?

    A[1,1]     = 0.5 * A[1,1]
    A[end,end] = 0.5 * A[end,end]

    B[1,1]     = 0.5 * B[1,1]
    B[end,end] = 0.5 * B[end,end]

    w = B * ones(n)

return z,w,A,B,C,D
end
