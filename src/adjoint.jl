#
#--------------------------------------#
export linsolve
export tpose
#--------------------------------------#
function linsolve(p,x,problem,solver,args...)
    lhs,rhs = problem(p,x)
    return solver(lhs,rhs,false,args...)
end
#--------------------------------------#
Zygote.@adjoint function linsolve(p,x,problem,solver,args...)
    lhs,rhs = problem(p,x)
    u = solver(lhs,rhs,false,args...)
    function fun(u̅)
        λ = solver(lhs,u̅,true,args...)
        _,drdp=pullback((pp)->resi(u,pp,x,problem),p)
        out = drdp(λ)[1]
        return (out,nothing,nothing,nothing,nothing,nothing,nothing,nothing,nothing)
    end
    return u,fun
end
#--------------------------------------#
function resi(u,p,x,problem)
    lhs, rhs = problem(p,x)
    return rhs .- lhs(u);
end
#--------------------------------------#

function tpose(J)
    if typeof(J)<:AbstractArray{<:AbstractArray{}}
        return broadcast(transpose,J)
    else
        return J'
    end
end
