#
#--------------------------------------#
export interpMat
#--------------------------------------#
"""
    Compute the Lagrange interpolation matrix from xi to xo
"""
function interpMat(xo,xi)
    
no = length(xo);
ni = length(xi);

a = ones(1,ni);
for i=1:ni
    for j=1:(i-1)  a[i]=a[i]*(xi[i]-xi[j]); end
    for j=(i+1):ni a[i]=a[i]*(xi[i]-xi[j]); end
end
a = 1 ./ a;

J = zeros(no,ni);
s = ones(1,ni);
t = ones(1,ni);
for i=1:no
    x = xo[i];
    for j=2:ni
        s[j]      = s[j-1]    * (x-xi[j-1]   );
        t[ni+1-j] = t[ni+2-j] * (x-xi[ni+2-j]);
    end
    J[i,:] = a .* s .* t;
end

return J
end
#--------------------------------------#
export semInterp
#--------------------------------------#
"""
    Get local coordinate from physical coordinates

    u: field to interpolate
    xp,yp: physical coordinate of point
    rxe,rye,sxe,sye: inverse jacobian
    xe,ye: physical coordinates in element
"""
function semInterp(u,xp,yp,rxe,rye,sxe,sye,zr,zs,xe,ye
                ,tol=1e-8,maxiter=50)

# start from closest GLL point
d2 = @. (xe-xp)^2 + (ye-yp)^2
imin = argmin(d2)
r = zr[imin[1]]
s = zs[imin[2]]

Js = []
Jr = []
i = 0
while true
    Jr = interpMat(r,zr)
    Js = interpMat(s,zs)

    if(i == maxiter) break; end

    x = ABu(Js,Jr,xe)
    y = ABu(Js,Jr,ye)
    dx = xp .- x
    dy = yp .- y

    if(sqrt(dx.^2+dy.^2)[1] < tol) break; end

    # invere Jacobian
    rx = ABu(Js,Jr,rxe)
    ry = ABu(Js,Jr,rye)
    sx = ABu(Js,Jr,sxe)
    sy = ABu(Js,Jr,sye)

    # Newton iteration
    dr = rx * dx + ry * dx # [dr] = [rx ry] * [dx]
    ds = sx * dy + sy * dy # [ds]   [sx sy]   [dy]

    # add Hessian components
    #dr +=
    #ds +=

    println("r=",r,", s=",s,", iter=",i)

    if(abs(r+dr[1])>1. || abs(s+ds[1])>1.)
        println("approximation went to shit")
        return NaN
    end

    # update
    r += dr[1]
    s += ds[1]
    i += 1

end

return ABu(Js,Jr,u)[1]
end
#--------------------------------------#
