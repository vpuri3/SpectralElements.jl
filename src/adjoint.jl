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
    lhs,rhs = problem(a)
    u = solver(lhs,rhs,false)
    function fun(u̅)
        λ = solver(lhs,u̅,true)
        _,drda=pullback((aa)->resi(u,aa,problem),a)
        return (drda(λ)[1],nothing)
    end
    return u,fun
end
#--------------------------------------#
function resi(u,a,problem)
    lhs, rhs = problem(a)
    return rhs .- lhs(u);
end
#--------------------------------------#
