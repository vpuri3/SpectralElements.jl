#
export pcg
#
# Conjugate gradient for Laplace solve
#
function pcg(b,A,M)

tol = 1e-8;
n = length(b);
itmax = n;
if(length(M)==0) M=Matrix(I,n,n); end;

x   = copy(b);
Ax  = A * x;
ra  = b - Ax;
ha  = 0;
hp  = 0;
hpp = 0;
rp  = 0;
rpp = 0;
u   = 0;
k   = 0;

while(norm(ra,Inf) > tol)
ha = M * ra; # preconditioner
k += 1;
if(k==itmax) println("warning: res:",norm(ra,Inf)); return x; end;
hpp = hp;
rpp = rp;
hp  = ha;
rp  = ra;
t   = sum(rp.*hp);
if(k==1); u = copy(hp);
else;     u = hp + (t / sum(rpp.*hpp)) * u;
end
Au = A * u; # operator
a = t / sum(u.*Au);
x = x + a * u;
ra = rp - a * Au;
end

println("CG iter: ",k, " res: ",norm(ra,Inf));

return x
end
