#
export jac
"""
 Computes Jacobian and its inverse of transformation

 x = x(r,s), y = y(r,s)

 J = [xr xs],  Jinv = [rx ry]\n
     [yr ys]          [sx sy]

 [Dx] * u = [rx sx] * [Dr] * u
 [Dy]     = [ry sy]   [Ds]

 ⟹
 [1 0] = [rx sx] *  [Dr] [x y]
 [0 1]   [ry sy]    [Ds]

 ⟹
                  -1
 [rx sx] = [xr yr]
 [ry sy]   [xs ys]

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
