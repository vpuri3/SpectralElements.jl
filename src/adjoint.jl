export linsolve

function linsolve(p, problem, solver)
    lhs, rhs = problem(p)
    return solver(lhs, rhs, false)
end

Zygote.@adjoint function linsolve(p, problem, solver)
    lhs, rhs = problem(p)
    u = solver(lhs, rhs, false)
    return u, function(u̅)
        _,gp=pullback((p)->g(u,p,problem),p)
        λ = solver(lhs, u̅, true)
        return (-gp(λ)[1],)
    end
end

function g(u,p,problem)
    lhs, rhs = problem(p)
    lhs(u).-rhs
end
