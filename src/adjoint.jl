#
#--------------------------------------#
export linsolve
#--------------------------------------#
function linsolve(a,problem,solver,mult,M,Qx,Qy)
    lhs,rhs = problem(a)
    return solver(lhs,rhs,false)
end
#--------------------------------------#
Zygote.@adjoint function linsolve(a,problem,solver,mult,M,Qx,Qy)
    lhs,rhs = problem(a)
    u = solver(lhs,rhs,false)
    function fun(u̅)
        λ = solver(lhs,u̅,true)
        _,drda=pullback((aa)->resi(u,aa,problem),a)
        out = drda(λ)[1]#gatherScatter(drda(λ.*mult.*M)[1],Qx,Qy)
        return (out,nothing,nothing,nothing,nothing,nothing,nothing)
    end
    return u,fun
end
#--------------------------------------#
function resi(u,a,problem)
    lhs, rhs = problem(a)
    return rhs .- lhs(u);
end
#--------------------------------------#
