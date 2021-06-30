#
#--------------------------------------#
export pcg
#--------------------------------------#
"""
 Preconditioned conjugate gradient

 We can't just use any packaged iterative solver for the PCG because
 we need to overwrite not just A*x, A'*x ops for Ax=b which is
 complicated enough since x,b  are multidimensional arrays (would have
 to allocate memory in converting and reconverting arrays to vecs), but
 also <u,v>, the dot product. because of the SEM construction, the dot
 product isnt just sum(u.*v) , but it's sum(u.*v.* SEM_mult)

"""
function pcg(b,opA
            ;opM=x->x           # preconditioner
            ,mult=ones(size(b)) # SEM mult
            ,ifv=false          # verbose flag
            ,tol=1e-8           # tolerance
            ,maxiter=length(b)) # maximum number of iterations

n = length(b);

x   = zero(b) # zero out initial condition
Ax  = zero(b)
ra  = b - Ax;
ha  = zero(b)
hp  = zero(b)
hpp = zero(b)
rp  = zero(b)
rpp = zero(b)
u   = zero(b)
k   = 0

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
export pcg!
#--------------------------------------#
function pcg!(x,b,opA
             ;opM=x->x           # preconditioner
             ,mult=ones(size(b)) # SEM mult
             ,ifv=false          # verbose flag
             ,tol=1e-8           # tolerance
             ,maxiter=length(b)) # maximum number of iterations

    x .= pcg(b,opA
            ;opM=opM
            ,mult=mult
            ,ifv=ifv
            ,tol=tol
            ,maxiter=maxiter)

return
end
#--------------------------------------#
