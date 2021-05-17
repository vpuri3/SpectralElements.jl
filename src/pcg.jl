#
#--------------------------------------#
export pcg
#--------------------------------------#
"""
 Preconditioned conjugate gradient
"""
function pcg(b,opA;opM=x->x     # preconditioner
            ,mult=ones(size(b)) # SEM mult
            ,ifv=false          # verbose flag
            ,tol=1e-8           # tolerance
            ,maxiter=length(b)) # maximum number of iterations

n = length(b);

x   = @. 0*b;
Ax  = @. 0*b;
ra  = b - Ax;
ha  = 0;
hp  = 0;
hpp = 0;
rp  = 0;
rpp = 0;
u   = 0;
k   = 0;

while(norm(ra,Inf) > tol)
ha = opM * ra # preconditinoer
#println("PCG iter: ",k,", res: ",norm(ra,2));
if(k==maxiter) println("warning: res:",norm(ra,Inf)); return x; end;
k  += 1;
hpp = hp;
rpp = rp;
hp  = ha;
rp  = ra;
t   = sum(rp.*hp.*mult);
if(k==1)
    u = copy(hp);
else
    u = hp+(t/sum(rpp.*hpp.*mult))*u;
end
Au = opA * u # operator
a = t / sum(u.*Au.*mult);
x = x + a * u;
ra = rp - a * Au;
end

if(ifv) println("PCG iter: ",k,", res: ",norm(ra,2)); end

return x
end
#--------------------------------------#
