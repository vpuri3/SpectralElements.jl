#
#--------------------------------------#
export linsolve
#--------------------------------------#
function linsolve(a,f,problem,solver)
    lhs,rhs = problem(a,f)
    return solver(lhs,rhs,false)
end
#--------------------------------------#
Zygote.@adjoint function linsolve(a,f,problem,solver)
    u = linsolve(a,f,problem,solver)
    function fun(u̅)
        _,gp=pullback((a)->g(u,a,f,problem),a)
        lhs,rhs = problem(a,u̅)
        λ  = solver(lhs,rhs,true)
        return (gp(λ)[1],nothing)
    end
    return u,fun
end
#--------------------------------------#
function g(u,a,f,problem)
    lhs, rhs = problem(a,f)
    return rhs .- lhs(u);
end
#--------------------------------------#
