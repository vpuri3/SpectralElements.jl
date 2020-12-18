#
#--------------------------------------#
export linsolve
#--------------------------------------#
function linsolve(a,problem,solver)
    lhs,rhs = problem(a)
    return solver(lhs,rhs,false)
end
#--------------------------------------#
Zygote.@adjoint function linsolve(a,problem,solver)
    u = linsolve(a,problem,solver)
    function fun(u̅)
        _,gp=pullback((a)->g(u,a,problem),a)
        lhs,rhs = problem(a)
        λ  = solver(lhs,u̅,true)
        return (gp(λ)[1],)
    end
    return u,fun
end
#--------------------------------------#
function g(u,a,problem)
    lhs, rhs = problem(a)
    return rhs .- lhs(u);
end
#--------------------------------------#
