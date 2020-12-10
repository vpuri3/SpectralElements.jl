#
#----------------------------------------------------------------------#
export pcgLapl
#----------------------------------------------------------------------#
"""
 Preconditioned conjugate gradient for lapl
"""
function pcgLapl(b,M,Qx,Qy,Dr,Ds,G11,G12,G22,mult)

tol = 1e-8;
n = length(b);
itmax = n;
#if(length(M)==0) M=Matrix(I,n,n); end;

x   = copy(b);
Ax  = lapl(x,M,Qx,Qy,Dr,Ds,G11,G12,G22);
ra  = b - Ax;
ha  = 0;
hp  = 0;
hpp = 0;
rp  = 0;
rpp = 0;
u   = 0;
k   = 0;

while(norm(ra,Inf) > tol)
ha = ra; # preconditioner
#ha = mass(ra,[],Bi,[],[]);
if(k==itmax) println("warning: res:",norm(ra,Inf)); return x; end;
k  += 1;
hpp = hp;
rpp = rp;
hp  = ha;
rp  = ra;
t   = sum(rp.*hp.*mult);
if(k==1); u = copy(hp);
else;     u = hp + (t / sum(rpp.*hpp.*mult)) * u;
end
Au = lapl(u,M,Qx,Qy,Dr,Ds,G11,G12,G22); # operator
a = t / sum(u.*Au.*mult);
x = x + a * u;
ra = rp - a * Au;
end

println("Lapl PCG iter: ",k,", res: ",norm(ra,2));

return x
end
#----------------------------------------------------------------------#
export pcg
#----------------------------------------------------------------------#
"""
 Preconditioned conjugate gradient
"""
function pcg(b,opA,mult)

tol = 1e-8;
n = length(b);
itmax = n;
#if(length(M)==0) M=Matrix(I,n,n); end;

x   = copy(b);
Ax  = opA(x);
ra  = b - Ax;
ha  = 0;
hp  = 0;
hpp = 0;
rp  = 0;
rpp = 0;
u   = 0;
k   = 0;

while(norm(ra,Inf) > tol)
ha = ra; # preconditioner
#ha = opM(ra);
if(k==itmax) println("warning: res:",norm(ra,Inf)); return x; end;
k  += 1;
hpp = hp;
rpp = rp;
hp  = ha;
rp  = ra;
t   = sum(rp.*hp.*mult);
if(k==1); u = copy(hp);
else;     u = hp + (t / sum(rpp.*hpp.*mult)) * u;
end
Au = opA(u); # operator
a = t / sum(u.*Au.*mult);
x = x + a * u;
ra = rp - a * Au;
end

#println("PCG iter: ",k,", res: ",norm(ra,2));

return x
end
#----------------------------------------------------------------------#
