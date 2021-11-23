#
Random.seed!(123)
using OrdinaryDiffEq, DiffEqOperators, LinearAlgebra

"""
GOALS

1. have defaultlinsolve use predefined ldiv! if available on that type
    - allow for `Diagonal`, `LinearAlgebra.I`

2. add misc shit in each Krylov iteration call:
    https://github.com/SciML/DiffEqBase.jl/issues/366

3. Make SEM.jl operators AbstractDiffEqOperators

"""

n = 10
x = Array(range(start=0,stop=1,length=n))
A = 0.01*Tridiagonal(-ones(n-1),2ones(n),-ones(n-1))
rn = (du, u, p, t) -> begin
    mul!(du, A, u)
end
u0 = rand(n)
prob = ODEProblem(ODEFunction(rn, jac_prototype=JacVecOperator{Float64}(rn, u0; autodiff=false)), u0, (0, 10.))
#=============================================#
# DiffEqBase/linear_nonlinear.jl
#=============================================#
"""
 OrdinaryDiffEq/test/interface/linear_nonlinear_tests.jl
"""

""" default """
#sol = solve(prob, QNDF(autodiff=false)) # linsolve=DefaultLinSolve()
#@show sol.retcode
""" GMRES """
#sol = solve(prob, QNDF(autodiff=false, linsolve=LinSolveGMRES()))
#@show sol.retcode
""" CG """
#sol = solve(prob, QNDF(autodiff=false, linsolve=LinSolveCG()))
#@show sol.retcode
#=============================================#
# LinearSolve.jl
#=============================================#

#=============================================#
"""
 OrdinaryDiffEq/test/interface/mass_matrix_tests.jl

 how to provide inverse to mass_matrix. where is mass_matrix being inverted???
 how to set preconditioner,
"""

#---------------------------------------------#
return
