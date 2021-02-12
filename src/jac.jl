#
export jac
"""
    Computes Jacobian and its inverse of transformation
    x = x(r,s), y = y(r,s)

    J = [xr xs],  Jinv = [rx ry]
        [yr ys]          [sx sy]
"""
function jac(x,y,Dr,Ds)

xr = ABu([],Dr,x); # dx/dr
xs = ABu(Ds,[],x);
yr = ABu([],Dr,y);
ys = ABu(Ds,[],y);

J  = @. xr * ys - xs * yr;
Ji = @. 1 / J;

rx = @.  Ji * ys;
ry = @. -Ji * xs;
sx = @. -Ji * yr;
sy = @.  Ji * xr;

return J,Ji,rx,ry,sx,sy
end
